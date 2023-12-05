from einops import rearrange, repeat
import torch
import triton
import triton.language as tl

# IC = K, OC = N
# in_feats: M, K
# kernel: N, K // 8 -> cast to N, K
# kernel.T: K // 8, N
# scaling: N, K // G
# scaling.T: K // G, N
# zeros: N, K // G // 8
# zeros.T: K // G // 8, N

@triton.jit
def matmul_kernel_simple(
        # Pointers to matrices
        a_ptr, qw_ptr, c_ptr,
        scales_ptr, zeros_ptr,

        dbg_qwpacked_ptr, dbg_qwunpacked_ptr, dbg_dequant_ptr,
        dbg_scales_ptr,
        dbg_unpacked_zeros_ptr,
        dbg_to_add_ptr,

        # Matrix dimensions
        M, N, K,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        QUANT_GROUP_SIZE: tl.constexpr
):
    """Kernel for computing the matmul C = A x qw (qweights). """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) 
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K) # (K,)
    qw_shifter = (offs_k % 8) * 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    for k in range(0, 1):
        a_offs = (k * BLOCK_SIZE_K) + (offs_am[:, None] * K + offs_k[None, :]) # (M, K)
        a = tl.load(a_ptr + a_offs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        qw_offs = (((k * BLOCK_SIZE_K) + offs_k[:, None]) // 8) * N + offs_bn[None, :] # (K, N)
        qw_packed = tl.load(qw_ptr + qw_offs) # (K, N)

        if pid == 0 and k == 0:
            # create K x N offsets 
            k_x_n = tl.arange(0, BLOCK_SIZE_K)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
            tl.store(dbg_qwpacked_ptr + k_x_n, qw_packed)

        # Without the broadcast, Triton will silently do the wrong thing
        qw_unpacked = (qw_packed >> qw_shifter[:,None]) & 0xF

        if pid == 0 and k == 0:
            # create K x N offsets 
            k_x_n = tl.arange(0, BLOCK_SIZE_K)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
            tl.store(dbg_qwunpacked_ptr + k_x_n, qw_unpacked)

        k_iters_per_quant_group = QUANT_GROUP_SIZE // BLOCK_SIZE_K
        grp_idx = k // k_iters_per_quant_group

        grp_row_off = N * grp_idx
        col_offs = offs_bn
        scales = tl.load(scales_ptr + grp_row_off + col_offs) # (N,)

        if pid == 0 and k == 0:
            tl.store(dbg_scales_ptr + tl.arange(0, BLOCK_SIZE_N), scales)

        zeros_row_off = grp_row_off // 8
        idx_within_packed = grp_idx % 8
        packed_zeros = tl.load(zeros_ptr + zeros_row_off + col_offs) # (N,)
        unpacked_zeros = (packed_zeros >> (idx_within_packed * 4)) & 0xF

        if pid == 0 and k == 0:
            tl.store(dbg_unpacked_zeros_ptr + tl.arange(0, BLOCK_SIZE_N), unpacked_zeros)

        dequantized = scales[None, :].to(tl.float32) * (qw_unpacked.to(tl.float32) - unpacked_zeros[None, :].to(tl.float32))
        if pid == 0 and k == 0:
            # create K x N offsets 
            k_x_n = tl.arange(0, BLOCK_SIZE_K)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
            tl.store(dbg_dequant_ptr + k_x_n, dequantized)
        to_add = tl.dot(a, dequantized.to(tl.float16))

        if pid == 0 and k == 0:
            # create M x N offsets
            m_x_n = tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
            tl.store(dbg_to_add_ptr + m_x_n, to_add)

        accumulator += to_add
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    stride_cm = N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_simple(
    a, qw, qzeros, scales
):
    N = 4096
    pack_num = 8
    group_size = 128

    batch_sz, K = a.shape
    assert qw.shape[1] == K // pack_num
    assert qw.shape[0] == N
    assert qzeros.shape == (N, K // group_size // pack_num)
    assert scales.shape == (N, K // group_size)

    M = batch_sz
    assert M == 128, "batch size must be 128"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid_1d = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    _qw = qw.T.contiguous()
    assert _qw.shape == (K // pack_num, N)
    _scales = scales.T.contiguous()
    assert _scales.shape == (K // group_size, N)
    _qzeros = qzeros.T.contiguous()
    assert _qzeros.shape == (K // group_size // pack_num, N)

    assert a.is_contiguous()
    assert c.is_contiguous()
    assert _qw.is_contiguous()
    assert _scales.is_contiguous()
    assert _qzeros.is_contiguous()


    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    dbg_qwpacked = torch.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), device=a.device, dtype=torch.int32)
    dbg_qwunpacked = torch.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), device=a.device, dtype=torch.int32)
    dbg_dequant = torch.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), device=a.device, dtype=torch.float32)
    dbg_scales = torch.zeros((BLOCK_SIZE_N,), device=a.device, dtype=_scales.dtype)
    dbg_unpacked_zeros = torch.zeros((BLOCK_SIZE_N,), device=a.device, dtype=_qzeros.dtype)
    dbg_to_add = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), device=a.device, dtype=torch.float32)

    matmul_kernel_simple[grid_1d](
        a_ptr=a, qw_ptr=_qw, c_ptr=c,
        scales_ptr=_scales, zeros_ptr=_qzeros,
        dbg_qwpacked_ptr=dbg_qwpacked,
        dbg_qwunpacked_ptr=dbg_qwunpacked,
        dbg_dequant_ptr=dbg_dequant,
        dbg_scales_ptr=dbg_scales,
        dbg_unpacked_zeros_ptr=dbg_unpacked_zeros,
        dbg_to_add_ptr=dbg_to_add,
        M=M, N=N, K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        QUANT_GROUP_SIZE=group_size)
    torch.cuda.synchronize()

    eq = lambda *tensors: all((tensors[0] == tensor).all() for tensor in tensors[1:])
    assert eq(dbg_qwpacked[:8,:], repeat(_qw[0,:BLOCK_SIZE_N], 'n -> 8 n'))
    assert eq(dbg_qwpacked[8:8+8,:], repeat(_qw[1,:BLOCK_SIZE_N], 'n -> 8 n'))

    assert eq(dbg_qwunpacked[0,:], (_qw[0,:BLOCK_SIZE_N] >> 4 * 0) & 0xF)
    assert eq(dbg_qwunpacked[1,:], (_qw[0,:BLOCK_SIZE_N] >> 4 * 1) & 0xF)
    assert eq(dbg_qwunpacked[2,:], (_qw[0,:BLOCK_SIZE_N] >> 4 * 2) & 0xF)
    assert eq(dbg_qwunpacked[7,:], (_qw[0,:BLOCK_SIZE_N] >> 4 * 7) & 0xF)
    assert eq(dbg_qwunpacked[8,:], (_qw[1,:BLOCK_SIZE_N] >> 4 * 0) & 0xF)

    assert eq(dbg_scales, _scales[0,:BLOCK_SIZE_N])
    assert eq(dbg_unpacked_zeros, (_qzeros[0,:BLOCK_SIZE_N] >> 0) & 0xF)

    expected_dequant = _scales[0,:BLOCK_SIZE_N].to(torch.float32) * (dbg_qwunpacked.to(torch.float32) - dbg_unpacked_zeros[None, :].to(torch.float32))
    assert eq(dbg_dequant, expected_dequant)

    to_add = torch.matmul(a[:BLOCK_SIZE_M,:BLOCK_SIZE_K], dbg_dequant.to(torch.float16))
    torch.testing.assert_close(to_add.to(torch.float32), dbg_to_add)

    print("all loop 1 tests passed")

    return c

# 
# script_dir = Path(__file__).parent.absolute()
# 
# import sys
# sys.path.append(script_dir)

from pathlib import Path
import awq_inference_engine as ie

batch_size = 40
dim = 4096
pack_num = 8
group_size = 128
base = Path("/home/vedantroy/Desktop/vllm/llm-awq/examples/debug")

inputs = torch.load(base / 'inputs.pt')
qweight = torch.load(base / 'qweight.pt')
scales = torch.load(base / 'scales.pt')
qzeros = torch.load(base / 'qzeros.pt')

assert inputs.shape == (batch_size, dim) and inputs.dtype == torch.float16
assert qweight.shape == (dim, dim // pack_num) and qweight.dtype == torch.int32
assert scales.shape == (dim, dim // group_size) and scales.dtype == torch.float16
assert qzeros.shape == (dim, dim // group_size // pack_num) and qzeros.dtype == torch.int32

if batch_size < 128:
    a = inputs
    a = torch.cat([a] * 4, dim=0)
    a = a[:128]
    assert a.shape[0] == 128
    inputs = a

# add more elemnts to inputs until batch_size is 128
# if batch_size < 128:
#     inputs = torch.cat([inputs] * 4, dim=0)
#     print(f"inputs.shape={inputs.shape}")
#     assert inputs.shape[0] == 160

# def print_dsc(t):
#     print(f"shape: {t.shape}, dtype: {t.dtype}, min={t.min()}, max={t.max()}")
# 
# print_dsc(inputs)
# print_dsc(qweight)
# print_dsc(scales)
# print_dsc(qzeros)

# inputs are both
# shape: torch.Size([40, 4096]), dtype: torch.float16, min=-1.4326171875, max=1.564453125
# shape: torch.Size([4096, 512]), dtype: torch.int32, min=-2147465099, max=2147448174
# shape: torch.Size([4096, 32]), dtype: torch.float16, min=0.0004451274871826172, max=0.00801849365234375
# shape: torch.Size([4096, 4]), dtype: torch.int32, min=-2107209592, max=2106107736

out = ie.gemm_forward_cuda(inputs, qweight, scales, qzeros, group_size, 8)
out2 = ie.gemm_forward_cuda2(inputs, qweight, scales, qzeros, group_size, 8)
assert out.shape == (inputs.shape[0], dim) and out.dtype == torch.float16

# torch.testing.assert_close(out, out2, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(out, out2)


print("doing triton matmul ...")
out3 = matmul_simple(
    a=inputs, qw=qweight, qzeros=qzeros, scales=scales
)
assert out3.shape == out2.shape
torch.testing.assert_close(out3, out2)