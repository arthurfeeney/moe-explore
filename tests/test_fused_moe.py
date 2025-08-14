import math
import torch
from moe_explore.triton_kernels.fused_moe import (
    fused_moe,
    FusedMoeParams
)
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import torch_grouped_matmul_gather_scatter, random_routing, random_groups
import pytest

@pytest.mark.parametrize(
    "num_tokens,num_experts,K,N,dtype", 
    [
        (10, 4, 128, 128, torch.bfloat16),
        (200, 4, 512, 512, torch.bfloat16),
        (200, 4, 512, 512, torch.float16),
    ]
)
def test_fused_moe(
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
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
        
    params = FusedMoeParams(
        weight,
        group_indices,
        None,
        False,
        False,
        num_tokens,
        1,
        None
    )
    
    out = fused_moe(input, params)
    ref = torch_grouped_matmul_gather_scatter(input, params)
    
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
def test_fused_moe_gather(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    K: int, 
    N: int,
    dtype: torch.dtype
):
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   

    params = FusedMoeParams(
        weight,
        p.group_indices,
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        None
    )
    
    out = fused_moe(input, params)
    ref = torch_grouped_matmul_gather_scatter(input, params)
    
    assert out.isfinite().all() and ref.isfinite().all()
    torch.testing.assert_close(out, ref)

@pytest.mark.parametrize(
    "num_tokens_times_topk,num_experts,topk,K,N,dtype", 
    [
        (20, 4, 2, 128, 128, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.float16),
    ]
)
def test_fused_moe_scatter(
    num_tokens_times_topk: int,
    num_experts: int,
    topk: int, 
    K: int, 
    N: int,
    dtype: torch.dtype
):
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens_times_topk, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    
    # this test is setup to implicitly assume that we have already routed `num_tokens`
    # to `num_tokens * topk`. The routing and scales are generated using the original
    # `num_tokens`, because it ends up reducing after the scales.
    num_tokens = num_tokens_times_topk // topk
    topk_scores, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    params = FusedMoeParams(
        weight,
        p.group_indices,
        p.indices,
        False,  
        True,
        num_tokens,
        topk,
        topk_scores
    )

    out = fused_moe(input, params)
    ref = torch_grouped_matmul_gather_scatter(input, params).to(dtype)
        
    assert out.isfinite().all() and ref.isfinite().all()
    # TODO: This atol is quite loose. default is 1e-5.
    rtol = 1.6e-2 if dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=rtol)