import itertools
import math
from dataclasses import dataclass
import pytest
import torch
from moe_explore.functional.topk_moe import (
    topk_moe_forward,
    topk_moe_torch
)
from moe_explore.params import MOEParams
from moe_explore.testing import random_mlp, random_topk_router, assert_close, random_interleaved_glu

test_params = [
    (128, 128, 256, "relu", 8, 2, torch.bfloat16),
    #(256, 1024, 1024, "gelu", 8, 2, torch.float16),
    #(999, 2048, 2048, "gelu", 8, 2, torch.float16),
    #(5, 2048, 2048, "gelu", 8, 2, torch.bfloat16),
    #(999, 2048, 2048, "gelu", 64, 8, torch.float16),
    #(999, 2048, 2048, "gelu", 64, 8, torch.float16),
    #(2011, 2048, 768, "silu", 64, 8, torch.float16),
    #(999, 2048, 2048, "gelu", 64, 8, torch.bfloat16),
    #(999, 2048, 2048, "gelu", 64, 8, torch.bfloat16),
    #(2011, 2048, 768, "silu", 64, 8, torch.bfloat16),
    # TODO: backward of glu not supported yet.
    #(2011, 2048, 768, "swiglu", 64, 8, torch.bfloat16),
    #(2011, 2048, 768, "geglu", 64, 8, torch.bfloat16),
    #(128, 128, 128, "swiglu", 64, 8, torch.float16),
    #(128, 128, 128, "geglu", 64, 8, torch.float16),
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
    torch.autograd.set_detect_anomaly(True)
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
    mlp_func = random_interleaved_glu if "glu" in activation else random_mlp
    mlp_params = mlp_func(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype)
    
    input.requires_grad = True
    mlp_params.weight1.requires_grad = True
    mlp_params.weight2.requires_grad = True
        
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
        mlp_params,
        num_experts,
        topk
    )

    output = topk_moe_forward(
        input,
        moe_params
    )

    output.sum().backward()
    actual_weight1_grad = mlp_params.weight1.grad.data.clone()
    actual_weight2_grad = mlp_params.weight2.grad.data.clone()
    actual_tokens_grad = input.grad.data.clone()

    mlp_params.weight1.grad.data.zero_()
    mlp_params.weight2.grad.data.zero_()
    input.grad.data.zero_()
    
    ref_output = topk_moe_torch(
        input,
        moe_params
    )
    
    ref_output.sum().backward()
    ref_weight1_grad = mlp_params.weight1.grad.data.clone()
    ref_weight2_grad = mlp_params.weight2.grad.data.clone()
    ref_tokens_grad = input.grad.data.clone()
    
    assert output.isfinite().all()
    assert ref_output.isfinite().all()
    assert actual_tokens_grad.isfinite().all()
    assert ref_tokens_grad.isfinite().all()
    assert actual_weight1_grad.isfinite().all()
    assert ref_weight1_grad.isfinite().all()
    assert actual_weight2_grad.isfinite().all()
    assert ref_weight2_grad.isfinite().all()
    
    assert_close(output, ref_output)
    assert_close(actual_tokens_grad, ref_tokens_grad)
    assert_close(actual_weight1_grad, ref_weight1_grad)
    assert_close(actual_weight2_grad, ref_weight2_grad)