from pathlib import Path
import torch
import awq_inference_engine as ie

# Set parameters
batch_size = 40
dim = 4096
pack_num = 8
group_size = 128
base = Path("/home/vedantroy/Desktop/vllm/llm-awq/examples/debug")

# Load data
inputs = torch.load(base / 'inputs.pt')
qweight = torch.load(base / 'qweight.pt')
scales = torch.load(base / 'scales.pt')
qzeros = torch.load(base / 'qzeros.pt')

# Ensure the shapes and data types are correct
assert inputs.shape == (batch_size, dim) and inputs.dtype == torch.float16
assert qweight.shape == (dim, dim // pack_num) and qweight.dtype == torch.int32
assert scales.shape == (dim, dim // group_size) and scales.dtype == torch.float16
assert qzeros.shape == (dim, dim // group_size // pack_num) and qzeros.dtype == torch.int32

# Adjust batch size to 128 if necessary
if batch_size < 128:
    a = torch.cat([inputs] * 4, dim=0)
    a = a[:128]
    inputs = a

dbg = torch.zeros((64, 32), device=inputs.device, dtype=torch.float16)
out2 = ie.gemm_forward_cuda2(inputs, qweight, scales, qzeros, dbg, group_size, 8)

def matmul_simple(
    a, qw, qzeros, scales
):
    N = 4096
    pack_num = 8
    group_size = 128

    M, K = a.shape
    # ASSUMPTION:
    # all quantization / packing is done along the channel dimension
    # (channels become lower-resolution, but the # of channels is the same)
    assert qw.shape == (N, K // pack_num)
    assert qzeros.shape == (N, K // group_size // pack_num)
    assert scales.shape == (N, K // group_size)

    print("dequantizing matrix ...")
    n_rows_to_dequant = 64
    K2 = 32
    dequant_matrix = torch.zeros((n_rows_to_dequant, K2), dtype=torch.float32, device=a.device)

    for row in range(n_rows_to_dequant):
        dequant_row = torch.zeros((K2, ), dtype=torch.float32, device=a.device)
        for col in range(K2):
            group_idx = col // group_size
            scale = scales[row][group_idx].to(torch.float32)
            qzero = qzeros[row][group_idx // pack_num]
            qweight = qw[row][col // pack_num] 

            assert col // group_size == 0
            assert scale == scales[row][0]
            assert qzero == qzeros[row][0]
            assert qweight in [qw[row][0], qw[row][1], qw[row][2], qw[row][3]]

            qzero_unpacked = ((qzero >> (4 * (group_idx % pack_num))) & 0xF).to(torch.float32)
            qweight_unpacked = ((qweight >> (4 * (col % pack_num))) & 0xF).to(torch.float32)
            dequant = scale * (qweight_unpacked - qzero_unpacked)
            dequant_row[col] = dequant
        dequant_matrix[row] = dequant_row
    torch.cuda.synchronize()
    print("finished dequantizing ...")
    return dequant_matrix

manual_dequant = matmul_simple(inputs, qweight, qzeros, scales)
assert dbg.shape == manual_dequant.shape

# print the first row of each
print("manual dequant")
print(manual_dequant[0,:])
print("dbg")
print(dbg[0,:])
diff = dbg[0,:] - manual_dequant[0,:]
# increase line width
# set scientific notation to off
torch.set_printoptions(sci_mode=False, linewidth=500)
diff = diff.reshape(4, 8)
print(diff)

print("===debug---")
# scale * (weight - zero)
print(f"qweight: {qweight[0][:4]}")
# ChatGPT's unpacking
# For 2074639241: [9, 8, 11, 7, 8, 10, 11, 7]
# For -1899591993: [7, 12, 6, 8, 6, 12, 14, 8]
# For -1468302726: [10, 7, 10, 7, 11, 7, 8, 10]
# For -1418090100: [12, 8, 9, 10, 9, 7, 11, 10]
print(f"qzeros: {qzeros[0][0]}")
# ChatGPT's unpacking
# For 1232762521: [9, 9, 6, 7, 10, 7, 9, 4] ​​
print(f"scales: {scales[0][0]}")
# s = 0.0014123916625976562

# for first 32, them vs me:
# s * (9 - 9) => 0 vs s * (9 - 9)
# s * (8 - 9) => -0.0014 vs s * (8 - 9)
# s * (11 - 9) => 0.0028 vs s * (11 - 6)
# s * (7 - 9) => -0.0028 (I am not doing this)

def unpack_int32_to_4bit(int32):
    return [(int32 >> (4 * i)) & 0xF for i in range(8)]