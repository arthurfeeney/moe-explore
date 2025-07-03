import torch

from .router_impl.topk_router import topk_router
from .expert_permute import get_token_indices, expert_input_permute, expert_output_permute
from .matmul_gather_scatter import matmul_gather_scatter

def topk_moe_naive_forward(
    input,
    router_weight,
    expert_weights1,
    expert_weights2,
    input_dim,
    num_experts,
    topk,
    activation
):
    topk_scores, topk_indices = topk_router(input, router_weight, topk)

    _, hidden_dim = input.shape
    orig_shape = input.shape
    x_flat = input.view(-1, hidden_dim)
    flat_expert_indices = topk_indices.view(-1)
    flat_expert_weights = topk_scores.view(-1, 1)

    topk_scores, topk_indices = topk_router(input, router_weight, topk)
    idxs, tokens_per_expert, token_idxs = get_token_indices(topk_indices, topk)

    expert_cache = torch.zeros_like(input)
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = input[exp_token_idxs]
        
        expert_out = expert_tokens @ expert_weights1[expert_id]
        expert_out = activation(expert_out)
        expert_out = expert_out @ expert_weights2[expert_id]

        # scale by scores and reduce
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).repeat(1, input.shape[-1]),
            expert_out,
            reduce='sum'
        )
    return expert_cache


def topk_moe_matmul_gather_scatter_forward(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    expert_weights1: torch.Tensor,
    expert_weights2: torch.Tensor,
    input_dim: int,
    num_experts: int,
    topk: int,
    activation
):
    topk_scores, topk_indices = topk_router(input, router_weight, topk)
    indices, group_indices, gather_scatter_indices = get_token_indices(topk_indices, topk, zero_prefix=True)

    h = matmul_gather_scatter(
        input, 
        expert_weights1, 
        group_indices,
        gather_indices=gather_scatter_indices,
    )
    h = activation(h)
    h = matmul_gather_scatter(
        h,
        expert_weights2,
        group_indices=group_indices,
        gather_indices=None,
        scatter_indices=gather_scatter_indices,
        scales=topk_scores.view(-1),
        scales_indices=indices.view(-1),
        output_rows=input.size(0)
    )
    return h


def topk_moe_group_gemm_forward(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    expert_weights1: torch.Tensor,
    expert_weights2: torch.Tensor,
    input_dim: int,
    num_experts: int,
    topk: int,
    activation
):
    topk_scores, topk_indices = topk_router(input, router_weight, topk)
    group_token = expert_input_permute(input, topk_indices, topk)
    group_token = group_gemm_fn(group_token, expert_weights1)
    group_token.tokens = activation(group_token.tokens)
    group_token = group_gemm_fn(group_token, expert_weights2)
    output = expert_output_permute(group_token, topk_scores, input.size())
    return output
