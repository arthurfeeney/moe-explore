import itertools
import math
from dataclasses import dataclass
import pytest
import torch
<<<<<<< HEAD
from moe_explore.functional.topk_moe import (
    topk_moe_forward,
    topk_moe_torch
=======
from moe_explore.functional.mlp import (
    moe_mlp_torch, 
    moe_mlp_grouped_gemm_fused,
    moe_mlp_grouped_gemm
)
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm
>>>>>>> ce57e96 (Remove non-interleaved glu)
)
from moe_explore.params import MOEParams
from moe_explore.testing import random_glu, random_mlp, random_topk_router, assert_close, random_interleaved_glu
        
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
    (2011, 2048, 768, "swiglu", 64, 8, torch.bfloat16),
    (2011, 2048, 768, "geglu", 64, 8, torch.bfloat16),
    (128, 128, 128, "swiglu", 64, 8, torch.float16),
    (128, 128, 128, "geglu", 64, 8, torch.float16),
]

@pytest.mark.parametrize(
    "seq_len,input_dim,hidden_dim,activation,num_experts,topk,dtype", test_params)
def test_topk_moe(
    seq_len,
    input_dim,
    hidden_dim,
    activation,
    num_experts,
    topk,
    dtype
):
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
        
    weight_func = random_interleaved_glu if "glu" in activation else random_mlp
        
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
        weight_func(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype),
        num_experts,
        topk
    )

    output = topk_moe_forward(
        input,
        moe_params
    )

    ref_output = topk_moe_torch(
        input,
        moe_params
    )

    assert ref_output.isfinite().all()
<<<<<<< HEAD
    assert output.isfinite().all()
    assert_close(ref_output, output)
=======
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
>>>>>>> ce57e96 (Remove non-interleaved glu)
