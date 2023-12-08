# 1. Build pure Pytorch implementation of kernel & test it against CUDA kernel
# 2. [ctx] Triton Kernel is 2 OOM off from CUDA kernel

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

def matmul_simple(
    a, qw, qzeros, scales
):
    N = 4096
    pack_num = 8
    group_size = 128

    M, K = a.shape
    assert qw.shape == (N, K // pack_num)
    assert qzeros.shape == (N, K // group_size // pack_num)
    assert scales.shape == (N, K // group_size)

    print("dequantizing matrix ...")
    n_rows_to_dequant = 4
    dequant_matrix = torch.zeros((n_rows_to_dequant, K), dtype=torch.float32, device=a.device)

    for row in range(n_rows_to_dequant):
        dequant_row = torch.zeros((K, ), dtype=torch.float32, device=a.device)
        for col in range(K):
            group_idx = col // group_size
            scale = scales[row][group_idx].to(torch.float32)
            qzero = qzeros[row][group_idx // pack_num]
            qweight = qw[row][col // pack_num] 

            qzero_unpacked = ((qzero >> (4 * (group_idx % pack_num))) & 0xF).to(torch.float32)
            qweight_unpacked = ((qweight >> (4 * (col % pack_num))) & 0xF).to(torch.float32)
            dequant = scale * (qweight_unpacked - qzero_unpacked)
            dequant_row[col] = dequant
        dequant_matrix[row] = dequant_row
    torch.cuda.synchronize()
    print("finished dequantizing ...")

    out = a.to(torch.float16) @ dequant_matrix.T.to(torch.float16)
    assert out.shape == (M, n_rows_to_dequant)
    return out


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


dbg = torch.zeros((64, 32), device=inputs.device, dtype=torch.float16)

out = ie.gemm_forward_cuda(inputs, qweight, scales, qzeros, group_size, 8)
out2 = ie.gemm_forward_cuda2(inputs, qweight, scales, qzeros, dbg, group_size, 8)
assert out.shape == (inputs.shape[0], dim) and out.dtype == torch.float16
torch.testing.assert_close(out, out2)

torch_repro = matmul_simple(
    a=inputs, qw=qweight, qzeros=qzeros, scales=scales
)
assert torch_repro.shape == (inputs.shape[0], 4)
actual = out[:, :4]
assert torch_repro.shape == actual.shape

print(torch_repro[1][3], torch_repro.to(torch.float16)[1][3], actual[1][3])

torch.testing.assert_close(torch_repro.to(torch.float16), actual)


# tweak settings so that all rows are printed on 1 lne
# torch.set_printoptions(linewidth=200, sci_mode=False)
# 
# print(torch_repro)
# print(actual)
# print(torch_repro - actual.to(torch.float32))
# # for item in out2[:, 0][:10]:
#     # print(item.item())
# # print(out2[:, 0][:10])
# # assert out3.shape == out2.shape