import math
import torch
from moe_explore.triton_kernels.m_grouped_gemm import (
    m_grouped_gemm,
    MGroupedGEMMParams
)
from moe_explore.functional.scale_and_reduce import scale_and_reduce
from moe_explore.functional.activation import activation
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import torch_grouped_matmul_gather_scatter, random_routing, random_groups, assert_close
import pytest
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

@pytest.mark.parametrize("num_tokens,num_experts,K,N,activation,dtype", [
    (10, 4, 128, 128, None, torch.bfloat16),
    (200, 4, 512, 512, None, torch.bfloat16),
    (200, 4, 512, 512, "gelu", torch.bfloat16),
    (200, 4, 512, 512, "silu", torch.bfloat16),
    (200, 4, 512, 512, "swiglu", torch.bfloat16),
    (200, 4, 512, 512, "geglu", torch.bfloat16),
    (1000, 16, 1024, 1024, "gelu", torch.bfloat16),
    (1000, 16, 1024, 1024, "grad_silu", torch.bfloat16),
    (1000, 16, 1024, 1024, "grad_gelu", torch.bfloat16),
    (1000, 16, 1024, 1024, "swiglu", torch.bfloat16),
    (16000, 16, 1024, 1024, "geglu", torch.bfloat16),
    (16000, 16, 1024, 1024, "geglu", torch.bfloat16),
    # TODO: Group size one is broken.
    #(1000, 1, 1024, 1024, "geglu", torch.bfloat16),
    (1000, 2, 300, 20, "gelu", torch.bfloat16),
    # Check sizes that need masking
    (1000, 2, 1000, 1000, "gelu", torch.float32),
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
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N)
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
        
    params = MGroupedGEMMParams(
        None,
        False,
        False,
        num_tokens,
        topk=1,
        scales=None,
        activation=activation
    )
    
    out = m_grouped_gemm(input, weight, group_indices, params)
    ref = torch_grouped_matmul_gather_scatter(input, weight, group_indices, params)
        
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)
    
@pytest.mark.parametrize("num_tokens,num_experts,KN,act,dtype", [
    (100, 4, 128, None, torch.bfloat16),
    (100, 4, 128, "silu", torch.bfloat16),
    (100, 4, 128, "gelu", torch.bfloat16),
])
def test_m_grouped_gemm_eye(
    num_tokens: int,
    num_experts: int,
    KN: int,
    act: str,
    dtype: torch.dtype
):
    r"""
    This checks distributions with known properties, so it's easier to use tighter bounds.
    """
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens, KN), dtype=dtype, device="cuda")
    weight = torch.eye(KN, dtype=torch.bfloat16, device="cuda")[None, :].repeat(num_experts, 1, 1)
    group_indices = random_groups(num_tokens, num_experts, device="cuda")   
    params = MGroupedGEMMParams(
        None,
        False,
        False,
        num_tokens,
        topk=1,
        scales=None,
        activation=act
    )
    out = m_grouped_gemm(input, weight, group_indices, params)
    target = activation(input, act)
    torch.testing.assert_close(out, target, atol=1e-6, rtol=1e-6)

@pytest.mark.parametrize("num_tokens,num_experts,K,N,activation,dtype", [
    (10, 4, 128, 128, None, torch.bfloat16),
    (10, 4, 128, 128, "gelu", torch.float16),
    (10, 4, 128, 128, "silu", torch.float16),
    (10, 4, 128, 128, "swiglu", torch.float16),
    (10, 4, 128, 128, "geglu", torch.float16),
])
def test_m_grouped_gemm_zeros(
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
    All of the activations, act(0) == 0, so we can just check exact equality with 0.
    """
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.zeros((num_experts, K, N), dtype=dtype, device="cuda")
    group_indices = random_groups(num_tokens, num_experts, device="cuda")   
    params = MGroupedGEMMParams(
        None,
        False,
        False,
        num_tokens,
        topk=1,
        scales=None,
        activation=activation
    )
    out = m_grouped_gemm(input, weight, group_indices, params)
    assert (out == 0).all()

parameters = "num_tokens,num_experts,topk,K,N,activation,dtype"
gather_scatter_test_cases = [
    (10, 4, 2, 128, 128, None, torch.bfloat16),
    (200, 4, 2, 512, 512, None, torch.bfloat16),
    (200, 4, 2, 512, 512, "gelu", torch.bfloat16),
    (200, 4, 2, 512, 512, "swiglu", torch.bfloat16),
    (200, 4, 2, 512, 512, "geglu", torch.bfloat16),
    (1000, 4, 2, 1000, 1000, "gelu", torch.float32),
]

@pytest.mark.parametrize(parameters, gather_scatter_test_cases)
def test_m_grouped_gemm_gather(
    num_tokens: int,
    num_experts: int,
    topk: int,
    K: int, 
    N: int,
    activation,
    dtype: torch.dtype
):
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
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        scales=None,
        activation=activation
    )
    
    out = m_grouped_gemm(input, weight, p.group_indices, params)
    ref = torch_grouped_matmul_gather_scatter(input, weight, p.group_indices, params)
    
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)

@pytest.mark.parametrize(parameters, gather_scatter_test_cases)
def test_m_grouped_gemm_scatter(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    K: int, 
    N: int,
    activation,
    dtype: torch.dtype
):
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
        p.indices,
        False,  
        True,
        num_tokens,
        topk=topk,
        scales=topk_scores,
        activation=activation
    )

    out = m_grouped_gemm(input, weight, p.group_indices, params)
    out = scale_and_reduce(out, params.scales, params.num_tokens, params.topk, out.size(-1))    
    ref = torch_grouped_matmul_gather_scatter(input, weight, p.group_indices, params)
                
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)
    
@pytest.mark.parametrize("num_tokens,num_experts,K,N,a_store_transpose,a_eval_transpose,b_store_transpose,b_eval_transpose", [
    (1000, 4, 512, 512, False, False, False, False),
    # Inputs are stored transposed, row major order
    (1000, 4, 512, 512, False, False, True, False),
    (1000, 4, 512, 512, True, False, False, False),
    (1000, 4, 512, 512, True, False, True, False),
    # Evaluate transpose in kernel
    (1000, 4, 512, 512, False, False, False, True),
    (1000, 4, 512, 512, False, True, False, False),
    (1000, 4, 512, 512, False, True, False, True),
])
def test_m_grouped_gemm_layouts(
    num_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    a_store_transpose: bool,
    a_eval_transpose: bool,
    b_store_transpose: bool,
    b_eval_transpose: bool,
):
    r"""
    We always assume that the batch dimension is first, but each matrix can be stored in 
    a combination of 1. transposed/non-transposed and 2. row/column major.
    - Inside the kernel, the majorness can be determined from the tensor strides, but whether the kernel
      should evaluate the matrix transposed or not need to be passed in as a parameter.
    This test checks that we can handle all of four combinations.
    """
    # default to non-transposed and row major
    input = torch.randn((num_tokens, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=torch.bfloat16, device="cuda") / math.sqrt(N)
    
    if a_store_transpose:
        input = input.t().contiguous()
    if b_store_transpose:
        weight = weight.permute(0, 2, 1).contiguous()
        
    if a_eval_transpose:
        input = torch.randn((K, num_tokens), dtype=torch.bfloat16, device="cuda")
        input = input.t()
    if b_eval_transpose:
        weight = torch.randn((num_experts, N, K), dtype=torch.bfloat16, device="cuda") / math.sqrt(N)
        weight = weight.permute(0, 2, 1)
        
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
    params = MGroupedGEMMParams(
        None,
        False,
        False,
        num_tokens,
        topk=1,
        scales=None,
        activation=None,
        is_a_transposed=a_store_transpose,
        is_b_transposed=b_store_transpose
    )
    
    out = m_grouped_gemm(input, weight, group_indices, params)
    ref = torch_grouped_matmul_gather_scatter(input, weight, group_indices, params)
    
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)
    
def test_te_grouped_linear():
    num_tokens = 1000
    num_experts = 16
    K = 128
    N = 256
    activation = None
    dtype = torch.bfloat16
    
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N)
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
    params = MGroupedGEMMParams(
        None,
        False,
        False,
        num_tokens,
        topk=1,
        scales=None,
        activation=activation
    )
    out = m_grouped_gemm(input, weight, group_indices, params)
    
    grouped_linear = GroupedLinear(num_experts, K, N, bias=False, params_dtype=dtype)
    
    
    print(grouped_linear.weight1.data[0, :5])
    
    for i in range(num_experts):
        getattr(grouped_linear, f"weight{i}").data[:] = weight[i].t()
    
    print(weight[0, 0, :5])
    print(grouped_linear.weight1.data[0, :5])
    
    m_splits = (group_indices[1:] - group_indices[:-1]).tolist()
    ref = grouped_linear(input, m_splits=m_splits, is_first_microbatch=None)
    
    assert out.isfinite().all() and ref.isfinite().all()
    assert_close(out, ref)
    
    
    