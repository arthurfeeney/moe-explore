import torch
from moe_explore.functional.m_grouped_mlp import m_grouped_mlp, torch_grouped_mlp
from moe_explore.testing import random_routing, assert_close
from moe_explore.expert_permute import get_token_indices
from moe_explore.functional.activation import activation as activation_func
import math
import pytest

test_string = "num_tokens,num_experts,topk,activation,dtype,atol,rtol"
test_params = [
    (1000, 16, 4, "relu", torch.float16, 5e-2, None),
    (1000, 16, 4, "silu", torch.float16, 2e-1, 2e-1),
    (1000, 16, 4, "relu", torch.bfloat16, 5e-1, 2e-1),
    (1000, 16, 4, "silu", torch.bfloat16, 2e-1, 2e-1),
    # Test with float32 since there are so many floating point operations.
    # Difficult to check tolerances with lower precisions.
    (1000, 16, 4, "silu", torch.float32, None, None),
    (1000, 16, 4, "relu", torch.float32, None, None),
    (2000, 32, 8, "silu", torch.float32, None, None),
]

@pytest.mark.parametrize(test_string, test_params)
def test_m_grouped_mlp(
    num_tokens: int,
    num_experts: int,
    topk: int,
    activation: str,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
):  
    tokens = torch.randn((num_tokens, 512), dtype=dtype, device="cuda")
    weight1 = torch.randn((num_experts, 512, 512), dtype=dtype, device="cuda") / math.sqrt(512)
    weight2 = torch.randn((num_experts, 512, 512), dtype=dtype, device="cuda") / math.sqrt(512)
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
    )  
    
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
    
    assert_close(output, ref)
    assert_close(actual_weight2_grad, ref_weight2_grad)
    # NOTE: These use pretty large tolerances because there's a lot of operations leading into this.
    # When using silu, only ~1% of elements are off by this much. With relu it's more accurate.
    assert_close(actual_tokens_grad, ref_tokens_grad, atol=atol, rtol=rtol)
    if dtype is not torch.bfloat16:
       assert_close(actual_weight1_grad, ref_weight1_grad, atol=atol, rtol=rtol)
