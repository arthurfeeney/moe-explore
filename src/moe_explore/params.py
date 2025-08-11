from dataclasses import dataclass
import math
from typing import Union, Callable, Optional
import torch

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

@dataclass
class ExpertMatmulParams:
    tokens: torch.Tensor
    weight: torch.Tensor
    group_indices: torch.Tensor
    token_to_group_indices: torch.Tensor
    scales: Optional[torch.Tensor]