from moe_explore.functional.m_grouped_gemm import m_grouped_gemm, torch_grouped_gemm
from moe_explore.testing import random_groups, random_routing,assert_close
from moe_explore.expert_permute import get_token_indices
from moe_explore.functional.scale_and_reduce import scale_and_reduce
import torch
import math

def test_m_grouped_gemm():
    num_tokens = 1000
    num_experts = 16
    topk = 4
    
    tokens = torch.randn((num_tokens, 128), dtype=torch.float16, device="cuda")
    weight = torch.randn((num_experts, 128, 128), dtype=torch.float16, device="cuda") / math.sqrt(128)
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
    
def test_m_grouped_gemm_gather():
    num_tokens = 1000
    num_experts = 16
    topk = 4
    
    tokens = torch.randn((num_tokens, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((num_experts, 128, 128), dtype=torch.bfloat16, device="cuda") / math.sqrt(128)
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=torch.bfloat16)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )  
    gather = True
    scatter = False
    activation = None
    
    tokens.requires_grad = True
    weight.requires_grad = True
    
    output = m_grouped_gemm(tokens, weight, p.group_indices, p.indices, gather, scatter, num_tokens, topk, activation)
    output.mean().backward()
    actual_weight_grad = weight.grad.data.clone()
    actual_tokens_grad = tokens.grad.data.clone()

    weight.grad.data.zero_()
    tokens.grad.data.zero_()
    
    ref = torch_grouped_gemm(tokens, weight, p.group_indices, p.indices, gather, scatter, num_tokens, topk, activation)
    ref.mean().backward()
    ref_weight_grad = weight.grad.data.clone()
    ref_tokens_grad = tokens.grad.data.clone()
    
    assert_close(output, ref)
    assert_close(actual_tokens_grad, ref_tokens_grad)
    assert_close(actual_weight_grad, ref_weight_grad)
    
    
def setup(func):
    # Using a seed so different `func` generate the same weights / tokens.
    torch.manual_seed(0)
    # This is using torch.autograd.grad to help with debugging.
    # It makes it a little easier to check intermediate gradients.
    
    num_tokens = 1000
    num_experts = 16
    topk = 4
    
    tokens = torch.randn((num_tokens * topk, 128), dtype=torch.float16, device="cuda")
    weight = torch.randn((num_experts, 128, 128), dtype=torch.float16, device="cuda") / math.sqrt(128)
    topk_scores, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=torch.float16)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )
    gather = False
    scatter = True
    activation = None
    
    topk_scores.requires_grad = True
    tokens.requires_grad = True
    weight.requires_grad = True

    output = func(tokens, weight, p.group_indices, p.indices, gather, scatter, num_tokens, topk, activation)
    loss1 = scale_and_reduce(output, topk_scores, num_tokens, topk, weight.size(-1)).sum()
    
    grads = torch.autograd.grad(loss1, [output, weight], retain_graph=True)
    grad_output, grad_weight1 = grads[0], grads[1]
    grad_output_weight = None #torch.autograd.grad(output.sum(), [weight])[0]
    
    return loss1, output, grad_output, grad_weight1, grad_output_weight


def test_m_grouped_gemm_scatter():
    loss1, output, grad_output, grad_weight1, grad_output_weight = setup(m_grouped_gemm)
    loss2, ref, grad_ref, grad_weight2, grad_ref_weight = setup(torch_grouped_gemm)

    assert abs(loss1.item() - loss2.item()) < 1e-3
    assert_close(output, ref)
    assert_close(grad_output, grad_ref)
    assert_close(grad_weight1, grad_weight2)