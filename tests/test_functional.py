import pytest
import torch
from moe.functional import (
    topk_moe_naive_forward, 
    topk_moe_group_gemm_forward,
    topk_moe_matmul_gather_scatter_forward
)

test_params = [
    ('cuda', torch.float16, 128, 128, 256, 1, 1, torch.nn.functional.gelu)
]

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
    input = torch.randn((seq_len, input_dim), device=device, dtype=dtype)
    router_weight = torch.randn((input_dim, num_experts), device=device, dtype=dtype)
    expert_weights1 = torch.randn((num_experts, input_dim, hidden_dim), device=device, dtype=dtype)
    expert_weights2 = torch.randn((num_experts, hidden_dim, input_dim), device=device, dtype=dtype)

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
    input = torch.randn((seq_len, input_dim), device=device, dtype=dtype)
    router_weight = torch.randn((input_dim, num_experts), device=device, dtype=dtype)
    expert_weights1 = torch.randn((num_experts, input_dim, hidden_dim), device=device, dtype=dtype)
    expert_weights2 = torch.randn((num_experts, hidden_dim, input_dim), device=device, dtype=dtype)


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
