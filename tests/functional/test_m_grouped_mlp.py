import torch
from moe_explore.functional.m_grouped_mlp import m_grouped_mlp, torch_grouped_mlp
from moe_explore.testing import random_routing, assert_close
from moe_explore.expert_permute import get_token_indices
from moe_explore.functional.activation import activation as activation_func
import math

def test_m_grouped_mlp():
    num_tokens = 1000
    num_experts = 16
    topk = 4
    
    tokens = torch.randn((num_tokens, 128), dtype=torch.float16, device="cuda")
    weight1 = torch.ones((num_experts, 128, 128), dtype=torch.float16, device="cuda") / math.sqrt(128)
    weight2 = torch.ones((num_experts, 128, 128), dtype=torch.float16, device="cuda") / math.sqrt(128)
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=torch.bfloat16)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )  
    activation = "silu"
    
    tokens.requires_grad = True
    weight1.requires_grad = True
    weight2.requires_grad = True
    
    output = m_grouped_mlp(tokens, weight1, weight2, p.group_indices, p.indices, num_tokens, topk, activation)
    output.sum().backward()
    actual_weight1_grad = weight1.grad.data.clone()
    actual_weight2_grad = weight2.grad.data.clone()
    actual_tokens_grad = tokens.grad.data.clone()
 
    weight1.grad.data.zero_()
    weight2.grad.data.zero_()
    tokens.grad.data.zero_()
  
    ref = torch_grouped_mlp(tokens, weight1, weight2, p.group_indices, p.indices, num_tokens, topk, activation)
    ref.sum().backward()
    ref_weight1_grad = weight1.grad.data.clone()
    ref_weight2_grad = weight2.grad.data.clone()
    ref_tokens_grad = tokens.grad.data.clone()
    
    print(actual_weight1_grad[:, 1])
    print(ref_weight1_grad[:, 1])
    
    assert_close(output, ref)
    assert_close(actual_weight2_grad, ref_weight2_grad)
    # NOTE: These use pretty large tolerances because there's a lot of operations leading into this.
    # Only ~1% of elemeents are off by this much, so I think it's okay.
    assert_close(actual_tokens_grad, ref_tokens_grad, atol=1e-1)
    assert_close(actual_weight1_grad, ref_weight1_grad, atol=2e-1)
