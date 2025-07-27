import torch
from torch import nn

def topk_router(*args, **kwargs):
    return softmax_topk_router(*args, **kwargs)

def softmax_topk_router(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int
):
    scores = nn.functional.softmax(input @ router_weight, dim=-1, dtype=torch.float32)
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_scores = topk_scores.to(input.dtype)
    return topk_scores, topk_indices

def topk_softmax_router(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int
):
    logits = input @ router_weight
    topk_scores, topk_indices = torch.topk(logits, k=topk, dim=-1, sorted=False)
    topk_scores = topk_scores.softmax(dim=-1)
    return topk_scores, topk_indices
