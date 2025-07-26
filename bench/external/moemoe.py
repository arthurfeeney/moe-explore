import torch
from momoe import MoMoE
from momoe import TopKRouter
from moe_explore.router import topk_router

def momoe_forward(
    input: torch.Tensor,
    router: TopKRouter,
    mlp: MoMoE,
):
    r"""
    MoMoE only supports SWIGLU-style MoEs.
    """
    pass

