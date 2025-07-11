import pytest
import torch
import math
from moe_explore.functional.forward import (
    topk_moe_naive_forward, 
    topk_moe_group_gemm_forward,
    topk_moe_matmul_gather_scatter_forward
)

test_params = [
    ('cuda', torch.float16, 128, 128, 256, 1, 1, torch.nn.functional.gelu),
    ('cuda', torch.float16, 256, 1024, 1024, 1, 1, torch.nn.functional.gelu),
    ('cuda', torch.float16, 999, 2048, 2048, 1, 1, torch.nn.functional.gelu)
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
def test_topk_moe_naive_forward(
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

    output = topk_moe_naive_forward(
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
def test_topk_moe_matmul_gather_scatter_forward(
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

    gg_output = topk_moe_matmul_gather_scatter_forward(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    ref_output = topk_moe_naive_forward(
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
def test_topk_moe_group_gemm_forward(
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

    gg_output = topk_moe_group_gemm_forward(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

    ref_output = topk_moe_naive_forward(
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
