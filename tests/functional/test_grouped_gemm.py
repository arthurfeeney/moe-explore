from moe_explore.functional.m_grouped_gemm import m_grouped_gemm, torch_grouped_gemm
from moe_explore.testing import random_groups, torch_grouped_matmul_gather_scatter, assert_close
import torch
import math

def test_grouped_gemm():
    num_tokens = 100
    num_experts = 8
    topk = 2
    
    tokens = torch.randn((num_tokens, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((num_experts, 128, 128), dtype=torch.bfloat16, device="cuda") / math.sqrt(128)
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
    permute_indices = None
    gather = False
    scatter = False
    activation = None
    
    tokens.requires_grad = True
    weight.requires_grad = True
    
    output = m_grouped_gemm(tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation)
    output.sum().backward()
    actual_weight_grad = weight.grad.data.clone()
    actual_tokens_grad = tokens.grad.data.clone()

    weight.grad.data.zero_()
    tokens.grad.data.zero_()
    
    ref = torch_grouped_gemm(tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation)
    ref.sum().backward()
    ref_weight_grad = weight.grad.data.clone()
    ref_tokens_grad = tokens.grad.data.clone()
    
    assert_close(output, ref)
    assert_close(actual_tokens_grad, ref_tokens_grad)
    assert_close(actual_weight_grad, ref_weight_grad)