import torch
from typing import Callable, Union
from moe_explore.params import MLPParams
try:
    from scattermoe.mlp import MLP, GLUMLP
    HAVE_SCATTERMOE = True
except ImportError:
    HAVE_SCATTERMOE = False

def make_scattermoe_mlp(input_size: int, hidden_size: int, num_experts: int, topk: int, activation: str):
    if "glu" in activation:
        return GLUMLP(
            input_size=input_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=topk,
            activation=torch.nn.SiLU()
        )
    else:
        return MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=topk,
            activation=torch.nn.SiLU()
        )

def scattermoe_forward( 
    input: torch.Tensor,
    router: Callable,
    router_params: torch.Tensor,
    mlp,
    topk: int
):
    topk_scores, topk_indices, router_logits = router(input, router_params)
    output = mlp(input, topk_scores, topk_indices) 
    return output, router_logits
