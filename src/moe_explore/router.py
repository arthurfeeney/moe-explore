import torch
from torch import nn

@torch.compile
def softmax(x):
    return nn.functional.softmax(x, dim=-1, dtype=torch.float32)

@torch.compile
def topk_router(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    softmax_before_topk: bool = True,
    normalize_routing: bool = False
):
    logits = input @ router_weight
    if softmax_before_topk:
        logits = softmax(logits)

    topk_scores, topk_indices = torch.topk(logits, k=topk, dim=-1, sorted=False)

    if not softmax_before_topk:
        topk_scores = softmax(topk_scores)

    if normalize_routing:
        topk_scores /= topk_scores.sum(dim=-1, keepdim=True)

    topk_scores = topk_scores.to(input.dtype)
    return topk_scores, topk_indices