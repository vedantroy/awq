
from pathlib import Path
import torch
import awq_inference_engine as ie

batch_size = 40
dim = 4096
pack_num = 8
group_size = 128
base = Path(__file__).parent / 'debug'

inputs = torch.load(base / 'inputs.pt')
qweight = torch.load(base / 'qweight.pt')
scales = torch.load(base / 'scales.pt')
qzeros = torch.load(base / 'qzeros.pt')

assert inputs.shape == (batch_size, dim) and inputs.dtype == torch.float16
assert qweight.shape == (dim, dim // pack_num) and qweight.dtype == torch.int32
assert scales.shape == (dim, dim // group_size) and scales.dtype == torch.float16
assert qzeros.shape == (dim, dim // group_size // pack_num) and qzeros.dtype == torch.int32

# add more elemnts to inputs until batch_size is 128
# if batch_size < 128:
#     inputs = torch.cat([inputs] * (128 // batch_size), dim=0)
#     print(f"inputs.shape={inputs.shape}")
#     assert inputs.shape[0] == 120

# From the CUDA kernel
#  in_feats: M, IC [float16]
#  kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
#  scaling_factors: IC // G, OC [float16]
#  zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
#  assume that batch_size < 16 for now

def print_dsc(t):
    print(f"shape: {t.shape}, dtype: {t.dtype}, min={t.min()}, max={t.max()}")

print_dsc(inputs)
print_dsc(qweight)
print_dsc(scales)
print_dsc(qzeros)

# inputs are both
# shape: torch.Size([40, 4096]), dtype: torch.float16, min=-1.4326171875, max=1.564453125
# shape: torch.Size([4096, 512]), dtype: torch.int32, min=-2147465099, max=2147448174
# shape: torch.Size([4096, 32]), dtype: torch.float16, min=0.0004451274871826172, max=0.00801849365234375
# shape: torch.Size([4096, 4]), dtype: torch.int32, min=-2107209592, max=2106107736

out = ie.gemm_forward_cuda(inputs, qweight, scales, qzeros, group_size, 8)
out2 = ie.gemm_forward_cuda2(inputs, qweight, scales, qzeros, group_size, 8)
assert out.shape == (batch_size, dim) and out.dtype == torch.float16

# torch.testing.assert_close(out, out2, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(out, out2)