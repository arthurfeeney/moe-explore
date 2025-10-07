import math
import torch
from moe_explore.triton_kernels.k_grouped_gemm import (
    k_grouped_gemm,
    KGroupedGEMMParams
)
from moe_explore.testing import random_groups
import pytest

@pytest.mark.parametrize("num_tokens,num_experts,K,N,activation,dtype", [
    (10, 4, 128, 128, None, torch.bfloat16),
    (200, 4, 512, 512, None, torch.bfloat16)
])
def test_k_grouped_gemm(
    num_tokens: int,
    num_experts: int,
    K: int, 
    N: int,
    activation,
    dtype: torch.dtype
):
    assert torch.cuda.is_available()
    tokens = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    grad_output = torch.randn((num_tokens, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
    params = KGroupedGEMMParams(
        None,
        False,
        False,
        num_tokens,
        topk=1,
        activation=activation
    )
    
    out = k_grouped_gemm(tokens, grad_output, group_indices, params)
    
    assert out.size() == (num_experts, K, N)
    assert out.isfinite().all()