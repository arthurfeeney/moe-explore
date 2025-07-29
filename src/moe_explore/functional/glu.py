import torch
from torch.profiler import record_function

from moe_explore.router import topk_router
from moe_explore.expert_permute import get_token_indices, expert_input_permute, expert_output_permute
from moe_explore.triton_kernels.grouped_mm_gather_scatter import grouped_mm_gather_scatter
from moe_explore.params import GLUParams, MOEParams

def moe_glu_torch(
    input,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUParams = params.expert_params
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
            exp_token_idxs = perm_to_group_indices.permute_indices[start_idx:end_idx]
            expert_tokens = input[exp_token_idxs]
            
            expert_gate = expert_tokens @ ep.gate_weight[expert_id]
            expert_gate = ep.activation(expert_gate)
            expert_up = expert_tokens @ ep.up_weight[expert_id]
            expert_out = (expert_gate * expert_up) @ ep.down_weight[expert_id]

            # scale by scores and reduce
            expert_out.mul_(flat_expert_weights[perm_to_group_indices.indices[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, input.shape[-1]),
                expert_out,
                reduce='sum'
            )
    return expert_cache

def moe_glu_grouped_gemm_fused(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUParams = params.expert_params
    topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
    perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)

    gate = grouped_mm_gather_scatter(
        input, 
        ep.gate_weight, 
        perm_to_group_indices.group_indices,
        gather_indices=perm_to_group_indices.permute_indices,
        autotune_mode=autotune_mode
    )
    gate = ep.activation(gate)

    up = grouped_mm_gather_scatter(
        input, 
        ep.up_weight, 
        perm_to_group_indices.group_indices,
        gather_indices=perm_to_group_indices.permute_indices,
        autotune_mode=autotune_mode
    )

    gated = gate * up

    down = grouped_mm_gather_scatter(
        gated,
        ep.down_weight,
        group_indices=perm_to_group_indices.group_indices,
        gather_indices=None,
        scatter_indices=perm_to_group_indices.permute_indices,
        scales=topk_scores.view(-1),
        scales_indices=perm_to_group_indices.indices.view(-1),
        topk=params.topk,
        output_rows=input.size(0),
        autotune_mode=autotune_mode
    )
    return down

def moe_glu_grouped_gemm(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUParams = params.expert_params
    topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
    group_token = expert_input_permute(input, topk_indices, num_experts=params.num_experts, topk=params.topk)
    
    gate = grouped_mm_gather_scatter(
            group_token.tokens, 
            ep.gate_weight, 
            group_token.group_indices,
            autotune_mode=autotune_mode
    )
    gate = ep.activation(gate)
    up = grouped_mm_gather_scatter(
            group_token.tokens, 
            ep.up_weight, 
            group_token.group_indices,
            autotune_mode=autotune_mode
    )
    gated = up * gate
    group_token.tokens = grouped_mm_gather_scatter(
            gated, 
            ep.down_weight, 
            group_token.group_indices,
            autotune_mode=autotune_mode
    )
    output = expert_output_permute(group_token, topk_scores, input.size())
    return output
