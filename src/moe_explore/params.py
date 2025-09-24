from moe_explore.functional.activation import Activation
from dataclasses import dataclass
from typing import Union, Optional
import torch

@dataclass
class FFNParams:
    pass

@dataclass
class MLPParams(FFNParams):
    weight1: torch.Tensor
    weight2: torch.Tensor
    activation: Activation
    
@dataclass
class SharedExpertParams(FFNParams):
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    activation: Activation
    
@dataclass
class RouterParams:
    pass
    
@dataclass
class TopkRouterParams(RouterParams):
    router_weight: torch.Tensor
    topk: int
    softmax_before_topk: bool = True
    normalize_routing: bool = False

@dataclass
class ErnieRouterParams(RouterParams):
    router_weight: torch.Tensor
    bias: torch.Tensor
    topk: int
    
@dataclass
class MOEParams:
    router_params: RouterParams
    expert_params: FFNParams
    num_experts: int
    topk: int
    shared_expert_params: Optional[SharedExpertParams] = None
    num_shared_experts: int = 0

@dataclass
class ExpertMatmulParams:
    tokens: torch.Tensor
    weight: torch.Tensor
    group_indices: torch.Tensor
    token_to_group_indices: torch.Tensor
    scales: Optional[torch.Tensor]