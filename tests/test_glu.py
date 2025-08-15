import math
import torch
from moe_explore.triton_kernels.glu import (
    glu,
    GLUParams
)
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import torch_grouped_glu, random_routing, random_groups, uniform_weight_init
import pytest

@pytest.mark.parametrize(
    "num_tokens,num_experts,K,N,dtype", 
    [
        (10, 4, 128, 128, torch.bfloat16),
        (200, 4, 512, 512, torch.bfloat16),
        (200, 4, 512, 512, torch.float16),
    ]
)
def test_glu_grouped(
    num_tokens: int,
    num_experts: int,
    K: int, 
    N: int,
    dtype: torch.dtype
):
    r"""
    This is essentially testing a M-grouped gemm. 
    It is not really testing part of an MoE, since it does no routing.
    If one had a hypothetical three-layer MLP, something like this could be the middle layer.
    """
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    gate_weight = uniform_weight_init((num_experts, K, N), dtype=dtype, device="cuda")
    up_weight = uniform_weight_init((num_experts, K, N), dtype=dtype, device="cuda")
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
        
    params = GLUParams(
        gate_weight,
        up_weight,
        group_indices,
        None,
        False,
        num_tokens,
        1,
        activation="silu"
    )

    out = glu(input, params)
    ref = torch_grouped_glu(input, params)
    
    assert out.isfinite().all() and ref.isfinite().all()
    torch.testing.assert_close(out, ref)
    
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,K,N,dtype", 
    [
        (10, 4, 2, 128, 128, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.float16),
    ]
)
def test_glu_grouped_gather(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    K: int, 
    N: int,
    dtype: torch.dtype
):
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    gate_weight = uniform_weight_init((num_experts, K, N), dtype=dtype, device="cuda")
    up_weight = uniform_weight_init((num_experts, K, N), dtype=dtype, device="cuda")
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   
    
    params = GLUParams(
        gate_weight,
        up_weight,
        p.group_indices,
        p.indices,
        True,
        num_tokens,
        topk,
        activation="silu"
    )
    
    out = glu(input, params)
    ref = torch_grouped_glu(input, params)
    
    assert out.isfinite().all() and ref.isfinite().all()
    torch.testing.assert_close(out, ref)