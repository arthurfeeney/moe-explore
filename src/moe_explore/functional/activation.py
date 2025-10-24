from sympy.core.numbers import phi_fixed
import torch
from enum import StrEnum
from typing import assert_never, Optional, Callable
import math

class Activation(StrEnum):
    NONE = "none"
    RELU = "relu"
    SILU = "silu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    GRAD_RELU = "grad_relu"
    GRAD_SILU = "grad_silu"
    GRAD_GELU = "grad_gelu"
    GRAD_SWIGLU = "grad_swiglu"
    GRAD_GEGLU = "grad_geglu"

@torch.compile
def grad_relu(x: torch.Tensor):
    out = torch.empty_like(x)
    out[x > 0] = 1.0
    out[x <= 0] = 0.0
    return out

@torch.compile
def grad_silu(x: torch.Tensor):
    sig_x = torch.sigmoid(x)
    return sig_x * (1 + x * (1 - sig_x))

@torch.compile
def grad_gelu(x: torch.Tensor):
    INV_SQRT_2 = 0.70710678118654752440
    INV_SQRT_2_PI = 0.3989422804
    phi = 0.5 * (1 + torch.erf(x * INV_SQRT_2))
    pdf = INV_SQRT_2_PI * torch.exp(-0.5 * x * x)
    return phi + x * pdf

@torch.compile
def swiglu(x: torch.Tensor):
    gate = x[..., 0::2]
    up = x[..., 1::2]
    return torch.nn.functional.silu(gate) * up

@torch.compile
def grad_swiglu(pre_activated: torch.Tensor):
    gate = pre_activated[..., 0::2]
    up = pre_activated[..., 1::2]
    return torch.stack([grad_silu(gate) * up, torch.nn.functional.silu(gate)], dim=-1).view(pre_activated.shape)
    
@torch.compile
def geglu(x: torch.Tensor):
    gate = x[..., 0::2]
    up = x[..., 1::2]
    return torch.nn.functional.gelu(gate) * up

@torch.compile
def grad_geglu(pre_activated: torch.Tensor):
    gate = pre_activated[..., 0::2]
    up = pre_activated[..., 1::2]
    return torch.stack([grad_gelu(gate) * up, torch.nn.functional.gelu(gate)], dim=-1).view(pre_activated.shape)
    
def activation(
    x: torch.Tensor, 
    act: Optional[Activation] = None,
    grad_out: Optional[torch.Tensor] = None
):
    if act is None or act == Activation.NONE:
        return x
    if act == Activation.RELU:
        return torch.nn.functional.relu(x)
    elif act == Activation.SILU:
        return torch.nn.functional.silu(x)
    elif act == Activation.GELU:
        return torch.nn.functional.gelu(x)
    elif act == Activation.SWIGLU:
        return swiglu(x)
    elif act == Activation.GEGLU:
        return geglu(x)
    elif act == Activation.GRAD_RELU:
        return grad_relu(x) * grad_out
    elif act == Activation.GRAD_SILU:
        return grad_silu(x) * grad_out
    elif act == Activation.GRAD_GELU:
        return grad_gelu(x) * grad_out
    elif act == Activation.GRAD_SWIGLU:
        return grad_swiglu(x) * grad_out.repeat_interleave(2, dim=-1)
    elif act == Activation.GRAD_GEGLU:
        return grad_geglu(x) * grad_out.repeat_interleave(2, dim=-1)
    assert_never(act)