from dataclasses import dataclass
from typing import Callable
import torch
from torch.profiler import record_function

from moe_explore.router import topk_router
from moe_explore.expert_permute import get_token_indices, expert_input_permute, expert_output_permute
from moe_explore.triton_kernels.fused_moe import fused_moe, FusedMoeParams
from moe_explore.params import MOEParams, MLPParams

def moe_mlp_torch(
    input,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    with record_function("router"):
        topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
        flat_expert_weights = topk_scores.view(-1, 1)
        #idxs, tokens_per_expert, token_idxs = get_token_indices(topk_indices, topk, num_experts)
        perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts)

    with record_function("moe"):
        expert_cache = torch.zeros_like(input)
        for expert_id, end_idx in enumerate(perm_to_group_indices.group_indices):
            start_idx = 0 if expert_id == 0 else perm_to_group_indices.group_indices[expert_id - 1]
            if start_idx == end_idx:
                continue
            exp_token_idxs = perm_to_group_indices.indices[start_idx:end_idx] // params.topk
            expert_tokens = input[exp_token_idxs]
            
            expert_out = expert_tokens @ ep.weight1[expert_id]
            expert_out = ep.activation(expert_out)
            expert_out = expert_out @ ep.weight2[expert_id]

            # scale by scores and reduce
            expert_out.mul_(flat_expert_weights[perm_to_group_indices.indices[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, input.shape[-1]),
                expert_out,
                reduce='sum'
            )
    return expert_cache

def moe_mlp_grouped_gemm_fused(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
    perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)

    up_params = FusedMoeParams(
        weight=ep.weight1,
        group_indices=perm_to_group_indices.group_indices,
        permute_indices=perm_to_group_indices.indices,
        gather=True,
        scatter=False,
        num_tokens=input.size(0),
        topk=params.topk,
        scales=None
    )

    h = fused_moe(
        input,
        up_params,
        autotune_mode=autotune_mode
    )
    h = ep.activation(h)

    down_params = FusedMoeParams(
        weight=ep.weight2, 
        group_indices=perm_to_group_indices.group_indices,
        permute_indices=perm_to_group_indices.indices,
        gather=False,
        scatter=True,
        num_tokens=input.size(0),
        topk=params.topk,
        scales=topk_scores
    )

    h = fused_moe(
        h,
        down_params,
        autotune_mode=autotune_mode
    )
    
    return h

def moe_mlp_grouped_gemm(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
    perm_to_group_indices = expert_input_permute(input, topk_indices, num_experts=params.num_experts, topk=params.topk)
    num_tokens = input.size(0) * params.topk
    
    up_params = FusedMoeParams(
        weight=ep.weight1,
        group_indices=perm_to_group_indices.group_indices,
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=params.topk,
        scales=None
    )
    up = ep.activation(fused_moe(
        perm_to_group_indices.tokens, 
        up_params,
        autotune_mode=autotune_mode
    ))

    down_params = FusedMoeParams(
        weight=ep.weight2,
        group_indices=perm_to_group_indices.group_indices,
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=params.topk,
        scales=None
    )
    down = fused_moe(
        up,
        down_params,
        autotune_mode=autotune_mode
    )
    
    perm_to_group_indices.tokens = down
    
    output = expert_output_permute(perm_to_group_indices, topk_scores, params.topk, input.size())
    return output