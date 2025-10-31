from torch._dynamo import output_graph
from moe_explore.functional.m_grouped_gemm import m_grouped_gemm, torch_grouped_gemm
from moe_explore.testing import random_groups, random_routing,assert_close
from moe_explore.expert_permute import get_token_indices
from moe_explore.functional.scale_and_reduce import scale_and_reduce
import torch
import math
import pytest

test_string = "num_tokens,num_experts,topk,activation,dtype"
test_params = [
    (1000, 16, 4, None, torch.bfloat16),
    (1000, 16, 4, None, torch.float16),
    (1000, 32, 8, None, torch.bfloat16),
    (1000, 32, 8, None, torch.float16),
]

@pytest.mark.parametrize(test_string, test_params)
def test_m_grouped_gemm(
    num_tokens: int,
    num_experts: int,
    topk: int,
    activation: str,
    dtype: torch.dtype,
):
    tokens = torch.randn((num_tokens, 512), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, 512, 512), dtype=dtype, device="cuda") / math.sqrt(512)
    group_indices = random_groups(num_tokens, num_experts, device="cuda")
    permute_indices = None
    gather = False
    scatter = False
    
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

@pytest.mark.parametrize(test_string, test_params)
def test_m_grouped_gemm_gather(
    num_tokens: int,
    num_experts: int,
    topk: int,
    activation: str,
    dtype: torch.dtype,
):  
    tokens = torch.randn((num_tokens, 512), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, 512, 512), dtype=dtype, device="cuda") / math.sqrt(512)
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
    )  
    gather = True
    scatter = False
    
    tokens.requires_grad = True
    weight.requires_grad = True
    
    output = m_grouped_gemm(tokens, weight, p.group_indices, p.indices, gather, scatter, num_tokens, topk, activation)
    output.sum().backward()
    actual_weight_grad = weight.grad.data.clone()
    actual_tokens_grad = tokens.grad.data.clone()

    weight.grad.data.zero_()
    tokens.grad.data.zero_()
    
    ref = torch_grouped_gemm(tokens, weight, p.group_indices, p.indices, gather, scatter, num_tokens, topk, activation)
    ref.sum().backward()
    ref_weight_grad = weight.grad.data.clone()
    ref_tokens_grad = tokens.grad.data.clone()
    
    assert_close(output, ref)
    assert_close(actual_tokens_grad, ref_tokens_grad)
    assert_close(actual_weight_grad, ref_weight_grad)
    
    
def setup(func, num_tokens: int, num_experts: int, topk: int, dtype: torch.dtype):
    # Using a seed so different `func` generate the same weights / tokens.
    torch.manual_seed(0)
    
    tokens = torch.randn((num_tokens * topk, 512), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, 512, 512), dtype=dtype, device="cuda") / math.sqrt(512)
    topk_scores, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
    )
    gather = False
    scatter = True
    activation = None
    
    topk_scores.requires_grad = True
    tokens.requires_grad = True
    weight.requires_grad = True

    output = func(tokens, weight, p.group_indices, p.indices, gather, scatter, num_tokens, topk, activation)
    loss1 = scale_and_reduce(output, topk_scores, num_tokens, topk, weight.size(-1)).sum()
    
    # This is using torch.autograd.grad to help with debugging.
    # It makes it a little easier to check intermediate gradients.
    grads = torch.autograd.grad(loss1, [output, weight], retain_graph=True)
    grad_output, grad_weight1 = grads[0], grads[1]
    
    return loss1, output, grad_output, grad_weight1

@pytest.mark.parametrize(test_string, test_params)
def test_m_grouped_gemm_scatter(
    num_tokens: int,
    num_experts: int,
    topk: int,
    activation: str,
    dtype: torch.dtype,
):
    loss1, output, grad_output, grad_weight1 = setup(m_grouped_gemm, num_tokens, num_experts, topk, dtype)
    loss2, ref, grad_ref, grad_weight2 = setup(torch_grouped_gemm, num_tokens, num_experts, topk, dtype)

    assert_close(output, ref)
    assert_close(grad_output, grad_ref)
    assert_close(grad_weight1, grad_weight2)