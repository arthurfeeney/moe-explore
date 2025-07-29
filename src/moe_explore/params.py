from dataclasses import dataclass
import math
from typing import Union, Callable
import torch

# We don't want individual modules for everything
# because it's annoying to pass weight around.
# Instead, everything is functional and things
# are organized in dataclasses.

@dataclass
class MLPParams:
    router_weight: torch.Tensor
    weight1: torch.Tensor
    weight2: torch.Tensor
    activation: Callable

@dataclass
class GLUParams:
    router_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    activation: Callable

@dataclass
class MOEParams:
    expert_params: Union[MLPParams, GLUParams]
    num_experts: int
    topk: int

def random_mlp(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=torch.randn,
):
    return MLPParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype) / math.sqrt(num_experts),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim),
        activation
    )

def random_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=torch.randn,
):
    return GLUParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype) / math.sqrt(num_experts),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim),
        activation
    )
