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
    
def grad_relu(x: torch.Tensor):
    out = torch.empty_like(x)
    out[x > 0] = 1.0
    out[x <= 0] = 0.0
    return out
    
def grad_silu(x: torch.Tensor):
    sig_x = torch.nn.functional.sigmoid(x)
    return sig_x * (1 + x * (1 - sig_x))#sig_x + x * sig_x * (1 - sig_x)
    
def swiglu(x: torch.Tensor):
    assert x.shape[-1] % 2 == 0
    gate = x[..., 0::2]
    up = x[..., 1::2]
    return torch.nn.functional.silu(gate) * up

def geglu(x: torch.Tensor):
    assert x.shape[-1] % 2 == 0
    gate = x[..., 0::2]
    up = x[..., 1::2]
    return torch.nn.functional.gelu(gate) * up
    
def activation(x: torch.Tensor, act: Optional[Activation] = None):
    if act is None or act == Activation.NONE:
        return x
    assert act in Activation, f"Invalid activation: {act}"
    if act == Activation.RELU:
        return torch.nn.functional.relu(x)
    if act == Activation.SILU:
        return torch.nn.functional.silu(x)
    elif act == Activation.GELU:
        return torch.nn.functional.gelu(x)
    elif act == Activation.SWIGLU:
        return swiglu(x)
    elif act == Activation.GEGLU:
        return geglu(x)
    elif act == Activation.GRAD_RELU:
        return grad_relu(x)
    elif act == Activation.GRAD_SILU:
        return grad_silu(x)
    #elif act == Activation.GRAD_GELU:
    #    return grad_gelu(x)
    assert_never(act)