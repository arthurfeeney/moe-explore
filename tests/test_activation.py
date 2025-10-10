import torch
import torch.nn.functional as F
from functools import partial
from moe_explore.functional.activation import activation
from moe_explore.triton_kernels.activation import TRITON_ACTIVATIONS
from moe_explore.testing import assert_close
import pytest
from typing import Callable
import triton
import triton.language as tl

def act_and_grad(x, act, grad_act):
    y = act(x)
    x_grad = grad_act(x)
    return x_grad, y

def torch_swiglu(x):
    return F.silu(x[..., 0::2]) * x[..., 1::2]

def torch_geglu(x):
    return F.gelu(x[..., 0::2]) * x[..., 1::2]

@triton.jit
def activation_kernel(x_ptr, y_ptr, ACTIVATION: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(x_ptr + offsets)
    y = ACTIVATION(x)
    y_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * y.shape[1] + tl.arange(0, y.shape[1])
    tl.store(y_ptr + y_offsets, y)

@pytest.mark.parametrize(
    "act, torch_act", [
        ("silu", F.silu),
        ("gelu", F.gelu),
        ("relu", F.relu),
        ("swiglu", torch_swiglu),
        ("geglu", torch_geglu),
    ])
def test_torch_activation(act: str, torch_act: Callable):
    x = torch.randn((32, 32, 32), device="cuda", dtype=torch.float32)
    x_grad, y = act_and_grad(x.clone(), partial(activation, act=act), partial(activation, act=f"grad_{act}"))
    
    x.requires_grad = True
    ref_y = torch_act(x)
    ref_y.sum().backward()
    ref_x_grad = x.grad.data.clone()

    assert_close(y, ref_y, atol=1e-6, rtol=1e-6)
    assert_close(x_grad, ref_x_grad, atol=1e-6, rtol=1e-6)
    
@pytest.mark.parametrize(
    "act, torch_act", [
        ("silu", F.silu),
        ("gelu", F.gelu),
        ("relu", F.relu),
        ("swiglu", torch_swiglu),
        ("geglu", torch_geglu),
    ])
def test_triton_activation(act: str, torch_act: Callable):
    x = torch.randn((32, 32), device="cuda", dtype=torch.float32)
    x.requires_grad = True
    
    # forward
    if "glu" in act:
        y = torch.zeros((32, 16), device="cuda", dtype=torch.float32)
    else:
        y = torch.zeros_like(x)
    activation_kernel[(1, 1)](x, y, TRITON_ACTIVATIONS[act], x.size(0), x.size(1))
    y_ref = torch_act(x)
    print(y[-1])
    print(y_ref[-1])
    assert_close(y, y_ref, atol=1e-6, rtol=1e-6)
    
    # backward
    # reference grad
    y_ref.sum().backward()
    x_grad_ref = x.grad.data.clone()
    x.grad.data.zero_()
    # actual grad
    x_grad = torch.zeros_like(x)
    activation_kernel[(1, 1)](x, x_grad, TRITON_ACTIVATIONS[f"grad_{act}"], x.size(0), x.size(1))
    assert_close(x_grad_ref, x_grad, atol=1e-6, rtol=1e-6)