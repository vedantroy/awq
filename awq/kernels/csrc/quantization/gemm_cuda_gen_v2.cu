// Inspired by NVIDIA's FasterTransformer
/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}

*/

#include "gemm_cuda_v2.h"
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define ASSERT_IF(cond, check) assert(!(cond) || (check))
#define FANCY_ASSERT_IF(cond, lhs, rhs) \
    do { \
        if ((cond) && !((lhs) rhs)) { \
            printf("Assertion failed: %d is wrong for %s\n", lhs, #lhs); \
            assert(0); \
        } \
    } while(0)

// Pack two half values.
static inline __device__ __host__ unsigned __pack_half2(const half x,
                                                        const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

__device__ __forceinline__ int make_divisible(int c, int divisor) {
  // Same thing as ceil(float(c) / float(divisor))
  return (c + divisor - 1) / divisor;
}

__host__ int divide_round_up(int c, int divisor) {
  // Same thing as ceil(float(c) / float(divisor))
  return (c + divisor - 1) / divisor;
}

__device__ __forceinline__ int divide_round_up_gpu(int c, int divisor){
  // Same thing as ceil(float(c) / float(divisor))
  return (c + divisor - 1) / divisor;
}


// w/ new kernel
// G = group_size (I believe its the quantization group size)
template <int G>
// Each block is 32 x 4 threads
// __launch_bounds__(128) tells compiler max 128 threads per block
__global__ void __launch_bounds__(128) 
  gemm2_forward_4bit_cuda_m128n64k32
  (
  int split_k_iters, 
  half* __restrict__ A,  // in_feats
  int* __restrict__ B,  // kernel
  half* __restrict__ scaling_factors, 
  int* zeros, 
  int M,  // num_in_feats (batch size?)
  int IC, // in_chan
  int OC,  // out_chan
  half* __restrict__ C // out_feats
  ) 
{
  static constexpr uint32_t ZERO = 0x0;

  // Initialized to all 0s (using a for loop for some reason?)
  // Stores floats which are converted to fp16 before being written to output tensor
  // The output loop tries to write all 64 values
  // However, some of the values might be greater than the # rows
  // Init / Output loop bounds are of form 4, 2, 8

  // Question: The fuck is the below doing
  // int row_offset = (((int)blockIdx_y) / j_factors1) * 128 + (threadIdx.y % 2) * 64 + ax0_0_2 * 16 + (local_id % 4) / 2 * 8 + ((int)threadIdx.x) / 4;

  // The kernel is setup s.t 64 elements are processed at a time
  // Very interesting:
  // - 16 mmas (of shape (16x16) x (16x8) = 16x8) are done per thread
  // - Each thread stores 4 elements per mma (32 (thds/warp) * 4 = 16 * 8)
  // - 16 * 4 = 64

  // Not sure if the below split even matters
  // It's basically just chunks of 4 floats
  // **Imagine the rows are split as well** (new-line is a separator)
  // 01 02 03 04   05 06 07 08   09 10 11 12   13 14 15 16
  // 01 02 03 04   05 06 07 08   09 10 11 12   13 14 15 16
  // 01 02 03 04   05 06 07 08   09 10 11 12   13 14 15 16
  // 01 02 03 04   05 06 07 08   09 10 11 12   13 14 15 16
  float C_warp[64];


  // k split shifts all threads to the right by 8 (# of splits) * 32 columns
  // but all threads are still always within 32 columns of each other

  // A matrix of size (128 x 32) (w/ 8 bytes of padding on the right hand side)
  // for block 0, split 0
  // warp 0:
  // thd 0
  //     (0..32.., 0) ->  (0..32, 0)
  // thd 1
  //     (0..32.., 8) ->  (0..32, 8)
  // thd 4
  //     (1..33.., 0) ->  (1..33, 0)
  // thd 31
  //     (7..39.., 24) ->  (7..39, 24)
  // warp 1:
  // thd 0
  //     (8..40.., 0) ->  (0..8, 0)
  // warp 3:
  // thd 0
  //     (24..56..32*3+24=120, 0) ->  (24..56..120, 0)
  // thd 31
  //     (31..63..32*3+31=127, 24) -> (31..63..127, 24)

  #define A_rows 128
  #define shared_stride (32 + 8)
  #define A_elems (A_rows * shared_stride)
  __shared__ half A_shared[A_elems];

   // for debugging, set everything to -1
   if (threadIdx.x == 0 && threadIdx.y == 0) {
     for (int i = 0; i < A_elems; i++) {
       A_shared[i] = __float2half(-1.0);
     }
   }
    __syncthreads();


  // A matrix of size (64 x 32) (w/ 8 bytes of padding on the right hand side)
  // for block 0, split 0
  // warp 0:
  //   thd 0: (0..32,0) -> (0..32, 0)
  //   thd 1: -> (0..32, 8)
  //   thd 4: -> (1..33, 0)
  // warp 3:
  //   thd 0: ->  (24..56, 0)
  //   thd 31: -> (24+7=31..56+7=63, 24)

  // block 1, split 0
  #define B_rows 64
  #define B_elems (B_rows * shared_stride)
  __shared__ half B_shared[64 * shared_stride];

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int i = 0; i < 64 * shared_stride; i++) {
      B_shared[i] = __float2half(-1.0);
    }
  }
  __syncthreads();
  
  // __shared__ half scaling_factors_shared[64];
  // __shared__ half zeros_shared[64];

  // # of blocks of 64 required to cover OC
  int j_factors1 = ((OC + 64 - 1) / 64);

  int gridWidthBlocks = divide_round_up_gpu(OC, 64);
  int gridHeightBlocks = divide_round_up_gpu(M, A_rows);
  int blocksPerGrid = gridHeightBlocks * gridWidthBlocks;
  int blockIdxInGrid = blockIdx.x % blocksPerGrid;
  int gridSplitIdx = blockIdx.x / blocksPerGrid;


  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((M + 128 - 1) / 128 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 128 - 1) / 128 * j_factors1);

  assert(j_factors1 == divide_round_up_gpu(OC, 64));
  assert(blockIdx_y == blockIdx.x % (divide_round_up_gpu(M, 128) * j_factors1));
  assert(blockIdx_z == blockIdx.x / (divide_round_up_gpu(M, 128) * j_factors1));

  assert(blockIdx_y == blockIdxInGrid);
  assert(blockIdx_z == gridSplitIdx);

  
  // 01 02 03 04 05 06 07 08 (used for 2 mmas)
  // 01 02 03 04 05 06 07 08 ^
  // 01 02 03 04 05 06 07 08 ^
  // 01 02 03 04 05 06 07 08 ^
  half A_shared_warp[32];

  // 01 02 03 04  05 06 07 08 (all mmas w/ warpCol = 0)
  // 01 02 03 04  05 06 07 08 (all mmas w/ warpCol = 1)
  half B_shared_warp[16];

  for (int i_0_3_init = 0; i_0_3_init < 4; ++i_0_3_init) {
    for (int j_0_4_init = 0; j_0_4_init < 2; ++j_0_4_init) {
      for (int i = 0; i < 8; ++i) {
        C_warp[((i_0_3_init * 16) + (j_0_4_init * 8)) + i] = 0.0;
      }
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride_A = 4 * 32 * 8 / 32;
  static constexpr int row_stride = 4 * 32 * 8 / 32;
  const int make_divisible_multipler = 128 / G;
  const int zeros_w = make_divisible(make_divisible(IC / G, 8), make_divisible_multipler) * make_divisible_multipler;
  const int sf_w = zeros_w * 8;

  // bool ld_zero_flag = (threadIdx.y * 32 + threadIdx.x) * 8 < 64;
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  int matrixRowsPerGridRow = A_rows;
  int gridRowIdx = blockIdxInGrid / gridWidthBlocks;
  // this might be an unnecessary var
  // 0, 128, 256, etc.
  int gridRowIdxAsMatrixRowIdx = gridRowIdx * matrixRowsPerGridRow;
  assert(gridRowIdxAsMatrixRowIdx % A_rows == 0);

  static constexpr int threadsPerWarp = 32;
  static constexpr int rowsPerWarp = 8; // (32 * 8) / 32
  static constexpr int threadsPerRow = threadsPerWarp / rowsPerWarp; // 4

  int warpIdx = threadIdx.y;

  // if block 0,
  // warp 0, thd 0..3 => row 0
  // warp 0, thd 4..7 => row 1
  // warp 0, thd 31 => row 7
  // warp 3, thd 31 => (3 * 8 + floor(7.75)) = 24 + 7 = row 31
  int ld_A_row = (
        gridRowIdxAsMatrixRowIdx 
        + warpIdx * rowsPerWarp 
        + threadIdx.x / threadsPerRow
  );
  ASSERT_IF(blockIdxInGrid == 0, (0 <= ld_A_row && ld_A_row <= 31));

  #define FIRST_BLOCK_FIRST_WARP (blockIdxInGrid == 0 && warpIdx == 0)
  FANCY_ASSERT_IF(FIRST_BLOCK_FIRST_WARP && (threadIdx.x == 0 || threadIdx.x == 1), ld_A_row, == 0);


  half* A_ptr = A
                + ld_A_row * IC
                + (threadIdx.x % threadsPerRow) * 8;

  // All threads have an initial offset 
  // that starts within the first 24 elements
  assert(((A_ptr - A) % IC) <= ((threadsPerRow - 1) * 8));
  
  int* B_ptr = B
            + (
              (blockIdxInGrid % gridWidthBlocks) * 64
              + warpIdx * rowsPerWarp
              + threadIdx.x / threadsPerRow
            ) * (IC / 8)
           + (threadIdx.x % threadsPerRow);
  
// Why * 1 in the above line?

  int sharedOffset = warpIdx * rowsPerWarp *  shared_stride
                      + (threadIdx.x / threadsPerRow) * shared_stride
                      + (threadIdx.x % threadsPerRow) * 8;
  half* A_shared_ptr = A_shared + sharedOffset;
  half* B_shared_ptr = B_shared + sharedOffset;
  

  int* zeros_ptr = zeros
                + (
                  (blockIdxInGrid % gridWidthBlocks) * 64
                  + warpIdx * rowsPerWarp
                  + threadIdx.x / threadsPerRow
                ) * zeros_w;
                // this term is zero
                // + (((int)threadIdx.x) % (32 / 8)) / G ;
  
  half* scaling_factors_ptr = scaling_factors
                            + ((int)threadIdx.y) * sf_w * 8
                            + (((int)threadIdx.x) / (32 / 8)) * sf_w
                            + (((int)blockIdx_y) % j_factors1) * (64) * sf_w
                            // this term is zero
                            + (((int)threadIdx.x) % (32 / 8)) * 8 / G;


  // Haotian: TBD, check, May 29 11:46 AM PST
  half* C_ptr = C 
              + blockIdx_z * M * OC        // blockIdx_z -> split_k dim
              + (((int)blockIdx_y) % j_factors1) * 64
              + (((int)threadIdx.y) / 2) * 32
              + (((int)threadIdx.x) % 4) * 2;

  // preload s.f. and zeros

  // 4096 / 32 = 128 (split input into 128 chunks of 32)
  // k_bound = 16 (chunks per split)

  // chunksPerSplit ??
  int k_bound = make_divisible(IC / 32, split_k_iters); // (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + gridSplitIdx >= IC) k_bound -= 1;

  // obvious, but include these anyway
  assert(k_bound * split_k_iters == IC / 32);
  ASSERT_IF(IC == 4096, k_bound == 16);
  
  // TODO (Haotian): load scales and zero points to smem

  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    // chunkIdx ??
    int k_0_0 = _k_0_0 * split_k_iters + gridSplitIdx;
    // 0, 8, ... (8 * 15) + 0  = 120
    // 1, 9, ... (8 * 15) + 1  = 121 
    // 7, 15, ... (8 * 15) + 7 = 127
    assert(IC - (k_0_0 * 32) >= 32);
    // todo, could be more general
    ASSERT_IF(
      gridSplitIdx == (split_k_iters - 1) && _k_0_0 == (k_bound - 1), 
      IC - (k_0_0 * 32) == 32
    );

    __syncthreads();
    // TODO: Haotian: Here we assume M % cta_M = 0.
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) 
    {
      int rowOffset = ax0_ax1_fused_0 * row_stride_A;
      FANCY_ASSERT_IF(FIRST_BLOCK_FIRST_WARP && (threadIdx.x == 0 || threadIdx.x == 1), ((ld_A_row + rowOffset) % 32), == 0);

      half* base = A;
      half* target = A_ptr + (rowOffset * IC) + (k_0_0 * 32);
      int offset = target - base;

      // if (base <= target && target < base + 24) {
        // printf("blk: %d, g_blk: %d, g_split: %d, w_idx: %d, t_idx: %d, k_0_0: %d, offset: %d\n", 
          // blockIdx.x, blockIdxInGrid, gridSplitIdx, warpIdx, threadIdx.x, k_0_0, target - base);
        // TODO: bugs / issues w/ below printf (offset is 0) (hypothesis = > 7 args in printf is bad, but no docs)
        // https://stackoverflow.com/questions/77550347/cuda-printf-outputs-incorrect-0-if-i-add-one-more-value
        // https://www.reddit.com/r/CUDA/comments/1841prx/does_cuda_printf_only_support_8_arguments/

        // printf("blk: %d, g_blk: %d, g_row: %d, g_split: %d, w_idx: %d, t_idx: %d, k_0_0: %d, offset: %d\n", 
        //   blockIdx.x, blockIdxInGrid, gridRowIdx, gridSplitIdx, warpIdx, threadIdx.x, k_0_0, target - base);
        // printf("offset: %d\n", target - base);
      // }

      ASSERT_IF(
          // 4096 - 8 = 4088
          (offset % IC) == 4088, 
          k_0_0 == 127 
          // next line is equivalent to above (last chunk in split + last split)
          && (gridSplitIdx == split_k_iters - 1 && _k_0_0 == k_bound - 1)
          // starting offset == 24
          && (A_ptr - A) % IC == 24
          // same as above
          // && (threadIdx.x % threadsPerRow == 3)
      );

      assert(rowOffset == 0 || rowOffset == 32 || rowOffset == 32 * 2 || rowOffset == 32 * 3);

      half* dest_target = A_shared_ptr + rowOffset * 40;
      int target_off = dest_target - A_shared;
      ASSERT_IF(warpIdx == 0 && threadIdx.x == 0, target_off == 0 +  rowOffset * 40);
      ASSERT_IF(warpIdx == 0 && threadIdx.x == 1, target_off == 8 +  rowOffset * 40);
      ASSERT_IF(warpIdx == 0 && threadIdx.x == 2, target_off == 16 + rowOffset * 40);

      // (128 thread * 4 iters * 16 bytes) / 2
      // uint4 = 16 bytes
      // half = 2 bytes
      *(uint4*)(A_shared_ptr + rowOffset * 40) = 
          (ld_A_row + rowOffset < M) 
          ? *(uint4*)(A_ptr + (rowOffset * IC) + (k_0_0 * 32)) 
          : make_uint4(0, 0, 0, 0);
    }

    // debugging
    if (true) {
      __syncthreads();
      // Ensure A is right-padded w/ -1
      if (threadIdx.x == 0 && threadIdx.y == 0) {
          for (int i = 0; i < A_rows; i++) {
            for (int j = 32; j < 40; j++) {
              assert(A_shared[i * 40 + j] == __float2half(-1.0));
            }
          }
      }

      // all blocks in the first grid row + first split, load the same elements
      // here, we're specifically looking @ the first chunk of the first split
      if (gridRowIdx == 0 && warpIdx == 0 && threadIdx.x == 0 && k_0_0 == 0) {
        // ensure A_shared is equal to the first 128x32 chunk of A
        // (w/ zero padding for out of bound rows)
        bool failed = false;
        for (int i = 0; i < min(M, A_rows); ++i) {
          for (int j = 0; j < 32; ++j) {
            if (A_shared[i * 40 + j] !=  A[i * IC + j]) {
              // printf("i: %d, j: %d, A_shared: %f, A: %f\n", i, j, __half2float(A_shared[i * 40 + j]), __half2float(A[i * IC + j]));
              failed = true;
            }
          }
        }
        // Ensure zero padding works
        for (int i = M; i < A_rows; ++i) {
          for (int j = 0; j < 32; ++j) {
            if (A_shared[i * 40 + j] != __float2half(0)) {
              // printf("i: %d, j: %d, A_shared: %f\n", i, j, __half2float(A_shared[i * 40 + j]));
              failed = true;
            }
          }
        }
        assert(!failed);
      }
    }


    int* zeros_ptr_local = zeros_ptr + k_0_0 * 32 / G / 8;
    half* scaling_factors_ptr_local = scaling_factors_ptr + k_0_0 * 32 / G;

    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * (32 / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
      int rowOff = ax0_ax1_fused_0 * row_stride;
      int B_loaded_current = *(B_ptr_local + rowOff * (IC / 8));
      int zeros_loaded = *(zeros_ptr_local + rowOff * zeros_w);
      zeros_loaded >>= ((k_0_0 * 32 / G) % 8) * 4;
      // & 0xF extracts the rightmost 4 bits
      float current_zeros = (float)(zeros_loaded & 0xF);
      half scaling_factors_loaded = *(scaling_factors_ptr_local + rowOff * sf_w);
      half B_loaded_fp16[8];
      #pragma unroll
      for (int ic_1 = 0; ic_1 < 8; ic_1++){
        float current_single_weight_fp = (float)(B_loaded_current & 0xF);
        half dequantized_weight = __float2half(__half2float(scaling_factors_loaded) * (current_single_weight_fp - current_zeros));
        // bitwise right shift by 4 bits
        B_loaded_current = B_loaded_current >> 4;
        B_loaded_fp16[ic_1] = dequantized_weight;  
      }
      // write back
      *(uint4*)(B_shared_ptr + rowOff * shared_stride) = *reinterpret_cast<uint4*>(B_loaded_fp16);
    }
    __syncthreads();

    if (true) {
      // ensure B is left-padded w/ -1
      if (threadIdx.x == 0 && threadIdx.y == 0) {
          for (int i = 0; i < 64; i++) {
            for (int j = 32; j < 40; j++) {
              assert(B_shared[i * 40 + j] == __float2half(-1.0));
            }
          }

          // int gridColIdx = blockIdxInGrid % gridWidthBlocks;
          // if (gridColIdx == 0 && k_0_0 == 0) {
          // }
      }

    }

    // Load values from shared memory (A_shared) to registers (A_shared_warp)
    // 8 loop iterations corresponds to 8 mma instructions
    // Watch the NVIDIA GTC 2020 talk for details
    // Key point = values are broadcast from one thread to other threads in the warp
    // (8 iters) * (128 bits) * (128 threads) / 16 (bits per fp16) = 8 * 128^2 / 16
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
           "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
           : "=r"(addr)
           : "l"((void *)(
            A_shared 
            + ((warpIdx & 1) * A_elems / 2)
            + (ax0_0 * A_elems / 8)
            + (k_0_1 * 16)
            + ((threadIdx.x % 16) * shared_stride) + ((threadIdx.x / 16) * 8)
         )));

          unsigned* aOff = (unsigned *)(A_shared_warp + (ax0_0 * 8));
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(aOff[0]), "=r"(aOff[1]), "=r"(aOff[2]), "=r"(aOff[3])
            : "r"(addr)
          );
        }
      }
      
      for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(
              // (&(B_shared[((((((int)threadIdx.y) >> 1) * 1280) + (ax0_0_1 * 640)) + (k_0_1 * 16))])) 
              // + ((((((int)threadIdx.x) >> 4) * 320) + ((((int)threadIdx.x) & 7) * 40)) + (((((int)threadIdx.x) & 15) >> 3) * 8))))
              B_shared
              + (threadIdx.y >> 1) * 1280
              + (ax0_0_1 * 640)
              + (k_0_1 * 16)
              + ((threadIdx.x >> 4) * 320)
              + ((threadIdx.x & 7) * 40)
              + (((threadIdx.x & 15) >> 3) * 8)
          )));

          unsigned* bOff = (unsigned *)(B_shared_warp + (ax0_0_1 * 8));
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(bOff[0]), "=r"(bOff[1]), "=r"(bOff[2]), "=r"(bOff[3])
            : "r"(addr)
          );
        }
      }
          
      for (int warpRow = 0; warpRow < 4; ++warpRow) {
        // Each loop iteration loads 4 unsigned (8 halfs)
        // `warpRow * 8` advances the pointer by 8 halfs
        unsigned* aSharedPtr = (unsigned *)(A_shared_warp + (warpRow * 8));
        for (int warpCol = 0; warpCol < 2; ++warpCol) {
      
            int cWarpBaseIndex = (warpRow * 16) + (warpCol * 8);
            unsigned* bSharedPtr = (unsigned *)(B_shared_warp + (warpCol * 8));
            float* cWarpPtr1 = (float *)(C_warp + cWarpBaseIndex);

            // https://www.reddit.com/r/CUDA/comments/qk9rbs/i_dont_understand_how_cuda_kernel_works_within/
            // tensor core instructions operate on an entire warp

            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions
            // The matrix multiply and accumulate operation has the following form:
            // D = A * B + C
            // where D and C are called accumulators and may refer to the same matrix
            // A = MxK, B=KxN, C,D=MxN

            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
            // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
            // - .m16n8k16 specifies M,N,K
            // - 1st matrix = row, 2nd = col
            // - D=f32, A=f16, B=f16, C=f32
            //     this matches up w/ A_shared_warp, B_shared_warp being `half` and C_warp being float

            // Slides
            // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf


            // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/Using_Inline_PTX_Assembly_In_CUDA.pdf
            // registers: f = .f32, r = .u32 (I guess 2 floats are packed into a 32 bit int?)
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(cWarpPtr1[0]), "=f"(cWarpPtr1[1]), "=f"(cWarpPtr1[2]), "=f"(cWarpPtr1[3])
                : "r"(aSharedPtr[0]), "r"(aSharedPtr[1]), "r"(aSharedPtr[2]), "r"(aSharedPtr[3]), 
                  "r"(bSharedPtr[0]), "r"(bSharedPtr[1]), 
                  "f"(cWarpPtr1[0]), "f"(cWarpPtr1[1]), "f"(cWarpPtr1[2]), "f"(cWarpPtr1[3])
            );

            unsigned* bSharedPtrOffset = bSharedPtr + 2;
            float* cWarpPtr2 = cWarpPtr1 + 4;

            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(cWarpPtr2[0]), "=f"(cWarpPtr2[1]), "=f"(cWarpPtr2[2]), "=f"(cWarpPtr2[3])
                : "r"(aSharedPtr[0]), "r"(aSharedPtr[1]), "r"(aSharedPtr[2]), "r"(aSharedPtr[3]), 
                  "r"(bSharedPtrOffset[0]), "r"(bSharedPtrOffset[1]), 
                  "f"(cWarpPtr2[0]), "f"(cWarpPtr2[1]), "f"(cWarpPtr2[2]), "f"(cWarpPtr2[3])
            );
        }
      }
    }
  }
    
// Haotian: Here (May 29 11:46AM PST)
// TODO: Shang: Hoist loop invariance.
  for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      for (int local_id = 0; local_id < 8; ++local_id) {
        int row_offset = (((int)blockIdx_y) / j_factors1) * 128 + (threadIdx.y % 2) * 64 + ax0_0_2 * 16 + (local_id % 4) / 2 * 8 + ((int)threadIdx.x) / 4;
        if (row_offset < M)
        {
          *(C_ptr + ax1_0 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax0_0_2 * 16) + (ax1_0 * 8) + local_id]);
        }
      }
    }
  }
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor gemm_forward_cuda2(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size,
    int split_k_iters)
{

    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // for int4, need _kernel.size(1) * 8

    // Some sort of performance optimization, where we compute ... 1/8 th ?? of the output at a time (since it is matmuls)
    // and then sum them up at the end
    at::Tensor _out_feats = torch::empty({split_k_iters, num_in_feats, _kernel.size(0)}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    // blockIdx_x: i_factors[0] * j_factors[0]
    // blockIdx_y: i_factors[1] * j_factors[1]

    if (num_out_channels % 64 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 64");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    int j_factors1 = num_out_channels / 64 / 1;

    // # of blocks of 128 required to cover M * number of blocks of 64 required to cover OC * split_k_iters
    // # of threads = (M * OC) / ((128 * 64) / split_k_iters)
    dim3 num_blocks((num_out_feats + 128 - 1) / 128 * j_factors1 * split_k_iters);

    int exp_num_blocks = (num_out_feats + 128 - 1) / 128 * j_factors1 * split_k_iters;
    // spawn enough blocks s.t there's a thread for each row (128 threads / block) and thread for every 64 columns
    assert(exp_num_blocks == divide_round_up(num_out_feats, 128) * (num_out_channels / 64) * split_k_iters);
    printf("n_block: %d, n_thread: %d, n_out (no_k): %d, n_out: %d\n", exp_num_blocks, exp_num_blocks * 128, num_out_feats * num_out_channels, num_out_feats * num_out_channels * split_k_iters);

    printf("ksize: %d, %d\n", _kernel.size(0), _kernel.size(1));
    assert(_kernel.size(0) == num_in_channels 
            && _kernel.size(1) == num_out_channels / 8);

    
    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dim3 threads_per_block(32, 4);
    if (group_size == 128)
    {
      gemm2_forward_4bit_cuda_m128n64k32<128><<<num_blocks, threads_per_block>>>(
        split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
    // else if (group_size == 64)
    // {
    //   gemm2_forward_4bit_cuda_m128n64k32<64><<<num_blocks, threads_per_block>>>(
    //     split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, out_feats);
    // }
    else
    {
      throw std::invalid_argument("Group size temporarily not supported.");
    }
    return _out_feats.sum(0);
}