import itertools
import math
from dataclasses import dataclass
import pytest
import torch
from moe_explore.functional.mlp import (
    moe_mlp_torch, 
    moe_mlp_grouped_gemm_fused,
    moe_mlp_grouped_gemm
)
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm
)
from moe_explore.params import MOEParams
from moe_explore.testing import random_glu, random_mlp, random_topk_router, assert_close
        
test_params = [
    (128, 128, 256, "gelu", 8, 2, torch.float16),
    (256, 1024, 1024, "gelu", 8, 2, torch.float16),
    (999, 2048, 2048, "gelu", 8, 2, torch.float16),
    (5, 2048, 2048, "gelu", 8, 2, torch.bfloat16),
    (999, 2048, 2048, "gelu", 64, 8, torch.float16),
    (999, 2048, 2048, "gelu", 64, 8, torch.float16),
    (2011, 2048, 768, "silu", 64, 8, torch.float16),
    (999, 2048, 2048, "gelu", 64, 8, torch.bfloat16),
    (999, 2048, 2048, "gelu", 64, 8, torch.bfloat16),
    (2011, 2048, 768, "silu", 64, 8, torch.bfloat16),
]

@pytest.mark.parametrize(
    "seq_len,input_dim,hidden_dim,activation,num_experts,topk,dtype", test_params)
def test_moe_mlp(
    seq_len,
    input_dim,
    hidden_dim,
    activation,
    num_experts,
    topk,
    dtype
):
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
    moe_params = MOEParams(
        random_topk_router(
            num_experts,
            input_dim,
            topk,
            softmax_before_topk=True,
            normalize_routing=False,
            device="cuda",
            dtype=dtype
        ),
        random_mlp(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype),
        num_experts,
        topk
    )
    
    gg_fused_output = moe_mlp_grouped_gemm_fused(
        input,
        moe_params
    )

    gg_output = moe_mlp_grouped_gemm(
        input,
        moe_params
    )

    ref_output = moe_mlp_torch(
        input,
        moe_params
    )

    assert ref_output.isfinite().all()
    assert gg_output.isfinite().all()
    assert gg_fused_output.isfinite().all()
    assert_close(ref_output, gg_output)
    assert_close(ref_output, gg_fused_output)

@pytest.mark.parametrize(
    "seq_len,input_dim,hidden_dim,activation,num_experts,topk,dtype", test_params)
def test_moe_glu(
    seq_len,
    input_dim,
    hidden_dim,
    activation,
    num_experts,
    topk,
    dtype
):
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
    moe_params = MOEParams(
        random_topk_router(
            num_experts,
            input_dim,
            topk,
            softmax_before_topk=True,
            normalize_routing=False,
            device="cuda",
            dtype=dtype
        ),
        random_glu(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype),
        num_experts,
        topk
    )

    gg_output = moe_glu_grouped_gemm(
        input,
        moe_params
    )

    ref_output = moe_glu_torch(
        input,
        moe_params
    )

    assert ref_output.isfinite().all()
    assert gg_output.isfinite().all()
    assert_close(ref_output, gg_output)
