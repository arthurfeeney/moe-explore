import torch
from torch import nn

def topk_router(
    input,
    router_weight,
    topk
):
    scores = (input @ router_weight).softmax(dim=-1)
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_scores, topk_indices
