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

  __shared__ half A_shared[128 * (32 + 8)];
  __shared__ half B_shared[64 * (32 + 8)];
  
  // __shared__ half scaling_factors_shared[64];
  // __shared__ half zeros_shared[64];

  // # of blocks of 64 required to cover OC
  int j_factors1 = ((OC + 64 - 1) / 64);

  int gridWidthBlocks = divide_round_up_gpu(OC, 64);
  int gridHeightBlocks = divide_round_up_gpu(M, 128);
  int blocksPerGrid = gridHeightBlocks * gridWidthBlocks;
  int blockIdxInGrid = blockIdx.x % blocksPerGrid;
  int gridIdx = blockIdx.x / blocksPerGrid;


  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((M + 128 - 1) / 128 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 128 - 1) / 128 * j_factors1);

  assert(j_factors1 == divide_round_up_gpu(OC, 64));
  assert(blockIdx_y == blockIdx.x % (divide_round_up_gpu(M, 128) * j_factors1));
  assert(blockIdx_z == blockIdx.x / (divide_round_up_gpu(M, 128) * j_factors1));

  
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
  // int ld_A_row = (blockIdx_y / j_factors1 * 128 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32);     // threadIdx.y is warp_id
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  int matrixRowsPerGridRow = 128;
  int gridRowIdx = blockIdxInGrid / gridWidthBlocks;
  // 0, 128, 256, etc.
  int matrixRowIdx = gridRowIdx * matrixRowsPerGridRow;

  static constexpr int threadsPerWarp = 32;
  static constexpr int rowsPerWarp = 8; // (32 * 8) / 32
  static constexpr int threadsPerRow = threadsPerWarp / rowsPerWarp;

  int warpIdx = threadIdx.y;

  // if block 0,
  // warp 0, thd 0..3 => row 0
  // warp 0, thd 4..7 => row 1
  // warp 0, thd 31 => row 7
  // warp 3, thd 31 => (3 * 8 + floor(7.75)) = 24 + 7 = row 31
  // int ld_A_row = (blockIdx_y / j_factors1 * 128 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32);     // threadIdx.y is warp_id
  int ld_A_row = (matrixRowIdx + warpIdx * rowsPerWarp + threadIdx.x / threadsPerRow);     // threadIdx.y is warp_id
  // if ((blockIdx_y / j_factors1 * 128) != matrixRowIdx) {
  //   printf("blockIdx_y: %d, j_factors1: %d, matrixRowIdx: %d, warpIdx: %d, threadIdx.x: %d, threadIdx.y: %d, ld_A_row: %d, ld_A_row_new: %d\n", blockIdx_y, j_factors1, matrixRowIdx, warpIdx, threadIdx.x, threadIdx.y, ld_A_row, ld_A_row_new);
  //   assert(false);
  // }
  // if ((threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) != (warpIdx * threadsPerWarp + threadIdx.x / threadsPerRow)) {
  //   printf("threadIdx.y: %d, row_stride_warp: %d, threadIdx.x: %d, threadsPerWarp: %d, threadIdx.x / threadsPerRow: %d, warpIdx: %d, threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32: %d, warpIdx * threadsPerWarp + threadIdx.x / threadsPerRow: %d\n", threadIdx.y, row_stride_warp, threadIdx.x, threadsPerWarp, threadIdx.x / threadsPerRow, warpIdx, threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32, warpIdx * threadsPerWarp + threadIdx.x / threadsPerRow);
  //   assert(false);
  // }


  half* A_ptr = A 
                + (((int)blockIdx_y) / j_factors1 * 128 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
                + (((int)threadIdx.x) % (32 / 8)) * 8;
  // half* A_ptr = A
  //               + (
  //                 matrixRowIdx
  //                 + warpIdx * rowsPerWarp
  //                 + threadIdx.x / threadsPerRow
  //               ) * IC
  //               // TODO: This is an annoying line
  //               // NO idea what it means
  //               + (threadIdx.x % threadsPerRow) * 8;
  
  int* B_ptr = B
            + ((int)threadIdx.y) * (IC / 8) * 8
            + (((int)threadIdx.x) / (32 / 8)) * (IC / 8)
            + (((int)blockIdx_y) % j_factors1) * 64 * (IC / 8)
            + (((int)threadIdx.x) % (32 / 8)) * 1;
  
// Why * 1 in the above line?
                        
  half* A_shared_ptr = A_shared 
                    + ((int)threadIdx.y) * row_stride_warp * (32 + 8) 
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                    + (((int)threadIdx.x) % (32 / 8) ) * 8;


  int row_index_in_shared = threadIdx.y * row_stride_warp;
  // Calculating the segment within the row, based on the x-coordinate of the thread
  int segment_within_row = threadIdx.x / (32 / 8);
  // Calculating the exact element offset within the segment
  int element_offset_within_segment = threadIdx.x % (32 / 8) * 8;

  // Computing the final pointer position in A_shared
  half* A_shared_ptr_2 = A_shared 
                        + row_index_in_shared * (32 + 8) 
                        + segment_within_row * (32 + 8)
                        + element_offset_within_segment;
  assert(A_shared_ptr == A_shared_ptr_2);

  half* B_shared_ptr = B_shared
                    + ((int)threadIdx.y) * (row_stride / 4) * (32 + 8)
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                    + (((int)threadIdx.x) % (32 / 8)) * 8;
  

  int* zeros_ptr = zeros
                + ((int)threadIdx.y) * zeros_w * 8
                + (((int)threadIdx.x) / (32 / 8)) * zeros_w
                + (((int)blockIdx_y) % j_factors1) * 64 * zeros_w
                // this term is zero
                + (((int)threadIdx.x) % (32 / 8)) / G ;
  
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
  int k_bound = make_divisible(IC / 32, split_k_iters); // (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx_z >= IC) k_bound -= 1;
  
  // TODO (Haotian): load scales and zero points to smem

  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    // TODO: Haotian: Here we assume M % cta_M = 0.
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) 
    {
      if (ld_A_row + ax0_ax1_fused_0 * row_stride_A < M)
      {
        *(uint4*)(A_shared_ptr + ax0_ax1_fused_0 * row_stride_A * 40) = *(uint4*)(A_ptr + (ax0_ax1_fused_0 * row_stride_A * IC) + (k_0_0 * 32));
      }
      else
      {
        *(uint4*)(A_shared_ptr + ax0_ax1_fused_0 * row_stride_A * 40) = make_uint4(0, 0, 0, 0);
      }
    }


    int* zeros_ptr_local = zeros_ptr + k_0_0 * 32 / G / 8;
    half* scaling_factors_ptr_local = scaling_factors_ptr + k_0_0 * 32 / G;

    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * (32 / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {

      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
      // row stride in shared memory: (NWARPS * 32 * 8 / cta_N) 
      int B_loaded_current = *(B_ptr_local + ax0_ax1_fused_0 * row_stride * (IC / 8));
      int zeros_loaded = *(zeros_ptr_local + ax0_ax1_fused_0 * row_stride * zeros_w);
      zeros_loaded >>= ((k_0_0 * 32 / G) % 8) * 4;
      float current_zeros = (float)(zeros_loaded & 0xF);
      half scaling_factors_loaded = *(scaling_factors_ptr_local + ax0_ax1_fused_0 * row_stride * sf_w);
      half B_loaded_fp16[8];
      #pragma unroll
      for (int ic_1 = 0; ic_1 < 8; ic_1++){
        float current_single_weight_fp = (float)(B_loaded_current & 0xF);
        half dequantized_weight = __float2half(__half2float(scaling_factors_loaded) * (current_single_weight_fp - current_zeros));
        B_loaded_current = B_loaded_current >> 4;
        B_loaded_fp16[ic_1] = dequantized_weight;  
      }
      // write back
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (32 + 8)) = *reinterpret_cast<uint4*>(B_loaded_fp16);
    }
    __syncthreads();
    // Load values from shared memory (A_shared) to registers (A_shared_warp)
    // 8 loop iterations corresponds to 8 mma instructions
    // Watch the NVIDIA GTC 2020 talk for details
    // Key point = values are broadcast from one thread to other threads in the warp
    // (8 iters) * (128 bits) * (128 threads) / 16 (bits per fp16) = 8 * 128^2 / 16
    // = 
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(
              (&(A_shared[((((((int)threadIdx.y) & 1) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])) + 
              (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
          );

          unsigned* aOff = (unsigned *)(A_shared_warp + (ax0_0 * 8));

          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(aOff[0]), "=r"(aOff[1]), "=r"(aOff[2]), "=r"(aOff[3])
            // : "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[3])
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
            : "l"((void *)((&(B_shared[((((((int)threadIdx.y) >> 1) * 1280) + (ax0_0_1 * 640)) + (k_0_1 * 16))])) + ((((((int)threadIdx.x) >> 4) * 320) + ((((int)threadIdx.x) & 7) * 40)) + (((((int)threadIdx.x) & 15) >> 3) * 8))))
          );
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[3])
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

                // Are these 2 lines the same?
                // float* cWarpPtr2 = (float *)(C_warp + cWarpBaseIndex + 4);
                // unsigned* bSharedPtrOffset = (unsigned *)(B_shared_warp + (warpCol * 8) + 4);

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

torch::Tensor gemm_forward_cuda2(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size,
    int split_k_iters)
{
    printf("ksize: %d, %d\n", _kernel.size(0), _kernel.size(1));

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