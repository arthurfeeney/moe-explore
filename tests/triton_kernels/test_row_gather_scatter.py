import pytest
import torch    
from moe_explore.triton_kernels.row_gather_scatter import row_gather, row_scatter

@pytest.mark.parametrize("in_row,out_row,dim,dtype", [
    (10, 10, 10, torch.float16), 
    (10, 100, 10, torch.float16),
    (100, 10, 10, torch.float16),
    (256, 256, 1024, torch.float32),
    (999, 1000, 99, torch.float32),
    (2, 47, 76, torch.float32),
])
def test_row_gather(in_row, out_row, dim, dtype):
    tokens = torch.randn(in_row, dim, device="cuda", dtype=dtype)
    gather_indices = torch.randint(0, in_row, (out_row,), device="cuda", dtype=torch.int32)
    output = row_gather(tokens, gather_indices)
    assert torch.allclose(output, tokens[gather_indices])

@pytest.mark.parametrize("in_row,dim,dtype", [
    (10, 10, torch.float16), 
    (10, 10, torch.float16),
    (100, 10, torch.float16),
    (256, 1024, torch.float32),
    (999, 99, torch.float32),
    (50, 76, torch.float32),
])
def test_row_permute(in_row, dim, dtype):
    tokens = torch.randn(in_row, dim, device="cuda", dtype=dtype)
    scatter_indices = torch.randperm(in_row, device="cuda", dtype=torch.int32)
    output = row_scatter(tokens, scatter_indices)
    ref_output = torch.zeros_like(tokens)
    ref_output[scatter_indices] = tokens
    assert torch.allclose(output, ref_output)
    
