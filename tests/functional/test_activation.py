import torch
from functools import partial
from moe_explore.functional.activation import activation
from moe_explore.testing import assert_close

def act_and_grad(x, act, grad_act):
    y = act(x)
    x_grad = grad_act(x)
    return x_grad, y

def test_activation():
    x = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    
    x_grad, y = act_and_grad(x.clone(), partial(activation, act="silu"), partial(activation, act="grad_silu"))
    
    x.requires_grad = True
    ref_y = torch.nn.functional.silu(x)
    ref_y.sum().backward()
    ref_x_grad = x.grad.data.clone()
    
    
    print(x_grad[0])
    print(ref_x_grad[0])
    
    assert_close(y, ref_y)
    assert_close(x_grad, ref_x_grad)    