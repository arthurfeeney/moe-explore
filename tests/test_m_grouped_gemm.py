import math
import torch
from moe_explore.triton_kernels.m_grouped_gemm import (
    m_grouped_gemm,
    MGroupedGEMMParams
)
from moe_explore.functional.scale_and_reduce import scale_and_reduce
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import torch_grouped_matmul_gather_scatter, random_routing, random_groups, assert_close
import pytest

@pytest.mark.parametrize("num_tokens,num_experts,K,N,activation,dtype", [
    (10, 4, 128, 128, None, torch.bfloat16),
    (200, 4, 512, 512, None, torch.bfloat16),
    (200, 4, 512, 512, "gelu", torch.bfloat16),
    (200, 4, 512, 512, "silu", torch.bfloat16),
    (200, 4, 512, 512, "swiglu", torch.bfloat16),
    (200, 4, 512, 512, "geglu", torch.bfloat16),
    (1000, 16, 1024, 1024, "gelu", torch.bfloat16),
    (1000, 16, 1024, 1024, "swiglu", torch.bfloat16),
    (16000, 16, 1024, 1024, "geglu", torch.bfloat16),
    # check weird sizes
    # TODO: Group size one is broken.
    #(1000, 1, 1024, 1024, "geglu", torch.bfloat16),
    (1000, 2, 300, 20, "gelu", torch.bfloat16),
])
def test_m_grouped_gemm(
    num_tokens: int,
    num_experts: int,
    K: int, 
    N: int,
    activation,
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
        
    print(group_indices)
        
    params = MGroupedGEMMParams(
        weight,
        group_indices,
        None,
        False,
        False,
        num_tokens,
        topk=1,
        scales=None,
        activation=activation
    )
    
    out = m_grouped_gemm(input, params)
    ref = torch_grouped_matmul_gather_scatter(input, params)
        
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)

parameters = "num_tokens,num_experts,topk,K,N,activation,dtype"
test_cases = [
    (10, 4, 2, 128, 128, None, torch.bfloat16),
    (200, 4, 2, 512, 512, None, torch.bfloat16),
    (200, 4, 2, 512, 512, "gelu", torch.bfloat16),
    (200, 4, 2, 512, 512, "swiglu", torch.bfloat16),
    (200, 4, 2, 512, 512, "geglu", torch.bfloat16),
]

@pytest.mark.parametrize(parameters, test_cases)
def test_m_grouped_gemm_gather(
    num_tokens: int,
    num_experts: int,
    topk: int,
    K: int, 
    N: int,
    activation,
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

    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        scales=None,
        activation=activation
    )
    
    out = m_grouped_gemm(input, params)
    ref = torch_grouped_matmul_gather_scatter(input, params)
    
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)

@pytest.mark.parametrize(parameters, test_cases)
def test_m_grouped_gemm_scatter(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    K: int, 
    N: int,
    activation,
    dtype: torch.dtype
):
    assert torch.cuda.is_available()
    num_tokens_times_topk = num_tokens * topk
    input = torch.randn((num_tokens_times_topk, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    
    # this test is setup to implicitly assume that we have already routed `num_tokens`
    # to `num_tokens * topk`. The routing and scales are generated using the original
    # `num_tokens`, because it ends up reducing after the scales.
    topk_scores, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        p.indices,
        False,  
        True,
        num_tokens,
        topk=topk,
        scales=topk_scores,
        activation=activation
    )

    out = m_grouped_gemm(input, params)
    out = scale_and_reduce(out, params.scales, params.num_tokens, params.topk, out.size(-1))    
    ref = torch_grouped_matmul_gather_scatter(input, params)
                
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)