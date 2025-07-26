import torch
from moe_explore.router import topk_router
from moe_explore.expert_permute import get_token_indices
from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm

@torch.compile
def get_m_sizes(topk_indices, topk, num_experts):
    indices = topk_indices.view(-1).argsort()
    m_sizes = torch.histc(indices, min=0, max=num_experts - 1, bins=num_experts)
    gather_indices = indices // topk
    return indices, m_sizes, gather_indices


def unsloth_forward(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    expert_weights1: torch.Tensor,
    expert_weights2: torch.Tensor,
    topk: int,
    num_experts: int,
    activation
):
    topk_scores, topk_indices = topk_router(input, router_weight, topk)
    indices, m_sizes, gather_indices = get_m_sizes(topk_indices, topk, num_experts)

    t = grouped_gemm(
        input,
        expert_weights1,
        m_sizes,
        topk,
        gather_indices,
        permute_x=True,
        permute_y=False,
    )
    t = activation(t)
    t = grouped_gemm(
        t,
        expert_weights2,
        m_sizes,
        topk,
        gather_indices,
        permute_x=False,
        permute_y=True,
    )
