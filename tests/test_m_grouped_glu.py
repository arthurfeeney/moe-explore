import math
import torch
from moe_explore.triton_kernels.m_grouped_glu import (
    m_grouped_glu,
    MGroupedGLUParams
)
from moe_explore.triton_kernels.m_grouped_glu_interleaved import (
    m_grouped_glu_interleaved,
    MGroupedGLUInterleavedParams
)
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import torch_grouped_glu, random_routing, random_groups, uniform_weight_init, assert_close
import pytest

@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,K,N,dtype", 
    [
        (10, 4, 2, 128, 128, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.float16),
    ]
)
def test_glu_interleaved(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    K: int, 
    N: int,
    dtype: torch.dtype
):
    assert torch.cuda.is_available()
    input = torch.randn((num_tokens * topk, K), dtype=dtype, device="cuda")
    gate_weight = uniform_weight_init((num_experts, K, N), dtype=dtype, device="cuda")
    up_weight = uniform_weight_init((num_experts, K, N), dtype=dtype, device="cuda")
    interleaved_weight = uniform_weight_init((num_experts, K, N * 2), dtype=dtype, device="cuda")
    interleaved_weight[:, :, 0::2] = gate_weight
    interleaved_weight[:, :, 1::2] = up_weight

    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    gemm_params = MGroupedGEMMParams(
        interleaved_weight,
        p.group_indices,
        None,
        False,
        False,
        num_tokens * topk,
        topk,
        scales=None,
        activation="geglu"
    )
    gemm_out = m_grouped_gemm(input, gemm_params)
    
    ref_params = MGroupedGLUParams(
        gate_weight,
        up_weight,
        p.group_indices,
        None,
        False,
        num_tokens,
        topk,
        activation="gelu"
    )
    ref = torch_grouped_glu(input, ref_params)
    
    assert gemm_out.isfinite().all() and ref.isfinite().all()
    assert_close(gemm_out, ref)

@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,K,N,dtype", 
    [
        (10, 4, 2, 128, 128, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.bfloat16),
        (2000, 32, 4, 512, 512, torch.float16),
    ]
)
def test_glu_interleaved_gather(
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
    interleaved_weight = torch.empty((num_experts, K, N * 2), dtype=dtype, device="cuda")
    interleaved_weight[:, :, 0::2] = gate_weight
    interleaved_weight[:, :, 1::2] = up_weight

    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   
        
    gemm_params = MGroupedGEMMParams(
        interleaved_weight,
        p.group_indices,
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        scales=None,
        activation="geglu"
    )
    gemm_out = m_grouped_gemm(input, gemm_params)
    
    ref_params = MGroupedGLUParams(
        gate_weight,
        up_weight,
        p.group_indices,
        p.indices,
        True,
        num_tokens,
        topk,
        activation="gelu"
    )
    ref = torch_grouped_glu(input, ref_params)
    
    assert gemm_out.isfinite().all() and ref.isfinite().all()
    assert_close(gemm_out, ref)