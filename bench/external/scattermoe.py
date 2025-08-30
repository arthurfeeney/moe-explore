import torch
from typing import Callable
from scattermoe.mlp import MLP

def scattermoe_forward(
    input: torch.Tensor,
    router: Callable,
    router_params: torch.Tensor,
    mlp: MLP,
    topk: int
):
    topk_scores, topk_indices = router(input, router_params)
    output = mlp(input, topk_scores, topk_indices) 
    return output
