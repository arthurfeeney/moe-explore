import torch
from moe_explore.router import topk_router
from scattermoe.mlp import MLP

def scattermoe_forward(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    mlp: MLP,
    topk: int
):
    topk_scores, topk_indices = topk_router(input, router_weight, topk)
    output = mlp(input, topk_scores, topk_indices) 
    return output
