import torch
import torch.nn.functional as F
from functools import partial
from moe_explore.functional.activation import activation
from moe_explore.testing import assert_close
import pytest
from typing import Callable

def act_and_grad(x, act, grad_act):
    y = act(x)
    x_grad = grad_act(x)
    return x_grad, y

@pytest.mark.parametrize(
    "act, torch_act", [
        ("silu", F.silu),
        ("gelu", F.gelu),
        ("relu", F.relu),
    ])
def test_activation(act: str, torch_act: Callable):
    x = torch.randn((32, 32, 32), device="cuda", dtype=torch.float32)
    x_grad, y = act_and_grad(x.clone(), partial(activation, act=act), partial(activation, act=f"grad_{act}"))
    
    x.requires_grad = True
    ref_y = torch_act(x)
    ref_y.sum().backward()
    ref_x_grad = x.grad.data.clone()
    
    assert_close(y, ref_y, atol=1e-6, rtol=1e-6)
    assert_close(x_grad, ref_x_grad, atol=1e-6, rtol=1e-6)