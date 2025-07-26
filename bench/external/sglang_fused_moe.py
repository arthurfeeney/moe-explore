import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe
)

def sglang_fused_moe(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    expert_weights1: torch.Tensor,
    expert_weights2: torch.Tensor,
    topk: int,
    activation,
):
    # This is for 0.4.9.post2, this API has been refactored on sglang's main branch
    # The fused MoE computes the topk and softmax on it's own.
    gating_output = input @ router_weight
    fused_moe(
        input,
        expert_weights1,
        expert_weights2,
        gating_output,
        topk=topk,
        renormalize=False,
        activation="gelu"
    )

