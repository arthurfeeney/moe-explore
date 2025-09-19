import torch
from enum import StrEnum
from typing import assert_never

class Activation(StrEnum):
    SILU = "silu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"

def activation(x: torch.Tensor, activation: Activation):
    assert activation in Activation, f"Invalid activation: {activation}"
    if activation == Activation.SILU:
        return torch.nn.functional.silu(x)
    elif activation == Activation.GELU:
        return torch.nn.functional.gelu(x)
    elif "glu" in activation:
        assert x.shape[-1] % 2 == 0
        gate = x[..., 0::2]
        up = x[..., 1::2]
        if activation == Activation.SWIGLU:
            gate = torch.nn.functional.silu(gate)
        elif activation == Activation.GEGLU:
            gate = torch.nn.functional.gelu(gate)
        return gate * up
    assert_never(activation)