import torch
from torch import nn
from moe_explore.params import RouterParams, TopkRouterParams, ErnieRouterParams

@torch.compile
def softmax(x):
    return nn.functional.softmax(x, dim=-1, dtype=torch.float32)

def router(input:torch.Tensor, params: RouterParams):
    if isinstance(params, TopkRouterParams):
        return topk_router(input, params)
    elif isinstance(params, ErnieRouterParams):
        return ernie_router(input, params)
    else:
        raise ValueError(f"Unsupported router type: {type(params)}")

@torch.compile
def topk_router(
    input: torch.Tensor,
    router_params: TopkRouterParams
):
    logits = input @ router_params.router_weight
    if router_params.softmax_before_topk:
        logits = softmax(logits)

    topk_scores, topk_indices = torch.topk(logits, k=router_params.topk, dim=-1, sorted=False)

    if not router_params.softmax_before_topk:
        topk_scores = softmax(topk_scores)

    if router_params.normalize_routing:
        topk_scores /= topk_scores.sum(dim=-1, keepdim=True)

    topk_scores = topk_scores.to(input.dtype)
    return topk_scores, topk_indices

@torch.compile
def ernie_router(
    input: torch.Tensor,
    router_params: ErnieRouterParams
):
    assert router_params.router_weight.dtype is torch.float32
    # Ernie uses float32 for everything in the router.
    with torch.autocast(device_type=input.device.type, enabled=False):
        logits = input.float() @ router_params.router_weight
        weights = softmax(logits)
        # in huggingface, it's topk(moe_statics(weights)), but the moe_statics is just adding a bias?
        _, topk_indices = torch.topk(weights + router_params.bias, k=router_params.topk, dim=-1, sorted=False)
        weights = torch.gather(weights, index=topk_indices, dim=-1)
        # The min=1e-12 is hardcoded from `moe_norm_min` in huggingface config.
        # I assume it's just to ensure there is not a division by zero.
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
        weights = weights.to(input.dtype)
        return weights, topk_indices