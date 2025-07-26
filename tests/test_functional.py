import pytest
import torch
import math
from moe_explore.functional.mlp import (
    moe_mlp_torch, 
    moe_mlp_grouped_gemm_fused,
    moe_mlp_grouped_gemm
)
from moe_explore.router import topk_router

test_params = [
    ('cuda', torch.float16, 128, 128, 256, 1, 1, torch.nn.functional.gelu),
    ('cuda', torch.float16, 256, 1024, 1024, 1, 1, torch.nn.functional.gelu),
    ('cuda', torch.float16, 999, 2048, 2048, 1, 1, torch.nn.functional.gelu),
    ('cuda', torch.bfloat16, 999, 2048, 2048, 1, 1, torch.nn.functional.gelu),
    ('cuda', torch.float32, 999, 2048, 2048, 1, 1, torch.nn.functional.gelu)
]

def generate_inputs(
    device,
    dtype,
    seq_len,
    input_dim,
    hidden_dim,
    num_experts
):
    input = torch.randn((seq_len, input_dim), device=device, dtype=dtype)
    router_weight = torch.randn((input_dim, num_experts), device=device, dtype=dtype) / math.sqrt(input_dim)
    expert_weights1 = torch.randn((num_experts, input_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(input_dim)
    expert_weights2 = torch.randn((num_experts, hidden_dim, input_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim)
    return (input, router_weight, expert_weights1, expert_weights2)

@pytest.mark.parametrize("device,dtype,seq_len,input_dim,hidden_dim,num_experts,topk,activation", test_params)
def test_moe_mlp_torch(
    device,
    dtype,
    seq_len,
    input_dim,
    hidden_dim,
    num_experts,
    topk,
    activation
):
    input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
            device, dtype, seq_len, input_dim, hidden_dim, num_experts)

    output = moe_mlp_torch(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    assert output.size() == input.size()

@pytest.mark.parametrize("device,dtype,seq_len,input_dim,hidden_dim,num_experts,topk,activation", test_params)
def test_moe_mlp_grouped_gemm(
    device,
    dtype,
    seq_len,
    input_dim,
    hidden_dim,
    num_experts,
    topk,
    activation
):
    input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
            device, dtype, seq_len, input_dim, hidden_dim, num_experts)

    gg_output = moe_mlp_grouped_gemm(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    ref_output = moe_mlp_torch(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    assert gg_output.size() == input.size()
    assert gg_output.size() == ref_output.size()
    assert torch.allclose(ref_output, gg_output, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("device,dtype,seq_len,input_dim,hidden_dim,num_experts,topk,activation", test_params)
def test_moe_mlp_grouped_gemm_fused(
    device,
    dtype,
    seq_len,
    input_dim,
    hidden_dim,
    num_experts,
    topk,
    activation
):
    input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
            device, dtype, seq_len, input_dim, hidden_dim, num_experts)

    gg_output = moe_mlp_grouped_gemm_fused(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    ref_output = moe_mlp_torch(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    assert gg_output.size() == input.size()
    assert gg_output.size() == ref_output.size()
    assert torch.allclose(ref_output, gg_output, atol=1e-2, rtol=1e-2)


