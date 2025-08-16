import torch
from enum import StrEnum
from typing import assert_never

class Activation(StrEnum):
    SILU = "silu"
    GELU = "gelu"

def activation(x: torch.Tensor, activation: Activation):
    assert activation in Activation, f"Invalid activation: {activation}"
    if activation == Activation.SILU:
        return torch.nn.functional.silu(x)
    elif activation == Activation.GELU:
        return torch.nn.functional.gelu(x)
    assert_never(activation)