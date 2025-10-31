import itertools
import math
from dataclasses import dataclass
import pytest
import torch
from moe_explore.functional.topk_moe import (
    topk_moe,
    topk_moe_torch
)
from moe_explore.router import router, topk_router
from moe_explore.params import MOEParams
from moe_explore.testing import random_mlp, random_topk_router, assert_close, random_interleaved_glu
from moe_explore.baseline.huggingface import make_huggingface_moe, set_huggingface_moe_weights
from moe_explore.baseline.torch_grouped_mm import TorchGroupedMMMoE
from moe_explore.gpu_utils import get_gpu_sm_version
#from moe_explore.baseline.transformer_engine import TransformerEngineMoE

test_params = [
    # This test runs the full forward and backward pass. With low precision and
    # larger weights, the errors accumulate by the time we're in the last part
    # of the backward pass. So, for lower precision, we only test smaller weights.
    (128, 512, 512, "relu", 8, 2, torch.float16),
    (128, 128, 128, "silu", 8, 2, torch.float16),
    (128, 512, 512, "relu", 8, 2, torch.float16),
    (128, 128, 128, "silu", 8, 2, torch.float16),
    (128, 128, 128, "gelu", 8, 2, torch.float16),
    (128, 128, 128, "swiglu", 8, 2, torch.float16),
    (128, 128, 128, "geglu", 8, 2, torch.float16),
    (128, 512, 512, "relu", 8, 2, torch.float16),
    # Run lots of tests in float32 since floating point errors don't accumulate as much.
    # Also checks lots of sizes that require masking.
    (999, 1024, 1024, "relu", 8, 2, torch.float32),
    (999, 1024, 1024, "silu", 8, 2, torch.float32),
    (1024, 1024, 1001, "gelu", 8, 2, torch.float32),
    (999, 1024, 1024, "swiglu", 8, 2, torch.float32),
    (999, 1000, 1000, "geglu", 8, 2, torch.float32),
    (999, 1000, 1000, "relu", 64, 8, torch.float32),
    (999, 1000, 1000, "silu", 64, 8, torch.float32),
    (999, 1000, 1000, "gelu", 64, 8, torch.float32),
    (999, 1000, 1000, "swiglu", 64, 8, torch.float32),
    (999, 1000, 1000, "geglu", 64, 8, torch.float32),
    (1, 1000, 1000, "geglu", 64, 8, torch.float32),
    (1, 30, 30, "geglu", 64, 8, torch.float32),
    # TODO: This case has an illegal memory acess???
    # (1, 1000, 1000, "geglu", 1, 1, torch.float32),
    (1, 30, 30, "geglu", 2, 1, torch.float32),
    (4000, 1000, 1000, "geglu", 1, 1, torch.float32),
    (4000, 1000, 1000, "geglu", 2, 1, torch.float32),
    (4000, 30, 30, "geglu", 4, 2, torch.float32),
    (4000, 1000, 1000, "geglu", 4, 2, torch.float32),
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
    # We need to rerun the full torch.compile for each input.
    torch._dynamo.reset()
    
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
    
    output = topk_moe(
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

def test_hf_moe():
    num_experts = 16
    seq_len = 128
    input_dim = 128
    hidden_dim = 256
    activation = "swiglu"
    topk = 4
    dtype = torch.bfloat16
    
    router_params = random_topk_router(
        num_experts,
        input_dim,
        topk,
        softmax_before_topk=True,
        normalize_routing=False,
        device="cuda",
        dtype=dtype
    )    
    mlp_params = random_interleaved_glu(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype)
    moe_params = MOEParams(
        router_params,
        mlp_params,
        num_experts,
        topk
    )
    
    mlp_params.weight1
    mlp_params.weight2
    
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
    
    output = topk_moe(
        input,
        moe_params
    )
    
    moe = make_huggingface_moe(input_dim, hidden_dim, num_experts, topk, activation, dtype).to("cuda").to(dtype)
    moe = set_huggingface_moe_weights(moe, router_params.router_weight, mlp_params.weight1, mlp_params.weight2)
    hf_hidden_states, hf_router_logits = moe(input.view(1, *input.size()))
    
    assert_close(output, hf_hidden_states.squeeze(0)) 
    
def test_torch_moe():
    num_experts = 16
    seq_len = 128
    input_dim = 128
    hidden_dim = 256
    activation = "swiglu"
    topk = 4
    dtype = torch.bfloat16
    
    router_params = random_topk_router(
        num_experts,
        input_dim,
        topk,
        softmax_before_topk=True,
        normalize_routing=False,
        device="cuda",
        dtype=dtype
    )    
    mlp_params = random_interleaved_glu(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype)
    moe_params = MOEParams(
        router_params,
        mlp_params,
        num_experts,
        topk
    )
    
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
    
    input.requires_grad = True
    moe_params.expert_params.weight1.requires_grad = True
    moe_params.expert_params.weight2.requires_grad = True
    
    output = topk_moe(input, moe_params)
    output.sum().backward()
    input_grad = input.grad.data.clone()
    weight1_grad = moe_params.expert_params.weight1.grad.data.clone()
    weight2_grad = moe_params.expert_params.weight2.grad.data.clone()
    
    input.grad.data.zero_()
    moe_params.expert_params.weight1.grad.data.zero_()
    moe_params.expert_params.weight2.grad.data.zero_()
    
    moe = TorchGroupedMMMoE(input_dim, hidden_dim, num_experts, topk, activation, dtype).to("cuda").to(dtype)
    moe.init_weights(moe_params.expert_params.weight1, moe_params.expert_params.weight2)
    moe.weight1.requires_grad = True
    moe.weight2.requires_grad = True
    moe.weight3.requires_grad = True
    ref, _ = moe(input, lambda x: router(x, router_params))
    ref.sum().backward()
    
    ref_input_grad = input.grad.data.clone()
    ref_weight1_grad = moe.weight1.grad.data.clone()
    ref_weight2_grad = moe.weight2.grad.data.clone()
    ref_weight3_grad = moe.weight3.grad.data.clone()
        
    ref_glu_grad = torch.empty_like(weight1_grad)
    ref_glu_grad[..., 0::2] = ref_weight1_grad
    ref_glu_grad[..., 1::2] = ref_weight2_grad
    
    assert_close(output, ref)
    assert_close(input_grad, ref_input_grad)
    assert_close(weight1_grad, ref_glu_grad)
    assert_close(weight2_grad, ref_weight3_grad)
    
"""
def test_te_moe():
    num_experts = 16
    seq_len = 128
    input_dim = 128
    hidden_dim = 256
    activation = "none"
    topk = 4
    dtype = torch.bfloat16
    
    router_params = random_topk_router(
        num_experts,
        input_dim,
        topk,
        softmax_before_topk=True,
        normalize_routing=False,
        device="cuda",
        dtype=dtype
    )    
    mlp_params = random_mlp(num_experts, input_dim, hidden_dim, activation, device="cuda", dtype=dtype)
    moe_params = MOEParams(
        router_params,
        mlp_params,
        num_experts,
        topk
    )
    
    input = torch.randn((seq_len, input_dim), device="cuda", dtype=dtype)
    
    output = topk_moe(
        input,
        moe_params
    )
    
    te_moe = TransformerEngineMoE(input_dim, hidden_dim, num_experts, topk, activation, dtype)
    te_moe.init_weights(mlp_params.weight1, mlp_params.weight2)
    te_output = te_moe(input, lambda x: router(x, router_params))
    
    assert_close(output, te_output)
"""