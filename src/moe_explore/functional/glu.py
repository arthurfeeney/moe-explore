import torch
from torch.profiler import record_function

from moe_explore.router import topk_router
from moe_explore.expert_permute import get_token_indices, expert_input_permute, expert_output_permute
from moe_explore.params import GLUParams, MOEParams
from moe_explore.triton_kernels.fused_moe import fused_moe, FusedMoeParams

import triton.profiler as proton

def moe_glu_torch(
    input,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
        flat_expert_weights = topk_scores.view(-1, 1)
    with proton.scope("get_token_indices"):
        #idxs, tokens_per_expert, token_idxs = get_token_indices(topk_indices, topk, num_experts)
        perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts)

    with proton.scope("moe"):
        expert_cache = torch.zeros_like(input)
        for expert_id, end_idx in enumerate(perm_to_group_indices.group_indices):
            with proton.scope(f"expert{expert_id}"):
                start_idx = 0 if expert_id == 0 else perm_to_group_indices.group_indices[expert_id - 1]
                if start_idx == end_idx:
                    continue
                exp_token_idxs = perm_to_group_indices.indices[start_idx:end_idx] // params.topk
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
    num_tokens = input.size(0)
    ep: GLUParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
    with proton.scope("get_token_indices"):
        perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)

    with proton.scope("moe"):
        gate_params = FusedMoeParams(
            ep.gate_weight,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=True,
            scatter=False,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=None
        )

        gate = fused_moe(
            input, 
            gate_params,
            autotune_mode=autotune_mode
        )
        gate = ep.activation(gate)

        up_params = gate_params
        up_params.weight = ep.up_weight

        up = fused_moe(
            input, 
            up_params,
            autotune_mode=autotune_mode
        )

        gated = gate * up

        down_params = FusedMoeParams(
            ep.down_weight,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=False,
            scatter=True,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=topk_scores
        )

        down = fused_moe(
            gated,
            down_params,
            autotune_mode=autotune_mode
        )

    return down

def moe_glu_grouped_gemm(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = topk_router(input, ep.router_weight, params.topk)
    with proton.scope("input_permute"):
        perm_to_group_indices = expert_input_permute(input, topk_indices, num_experts=params.num_experts, topk=params.topk)
        
    num_tokens = input.size(0) * params.topk
        
    with proton.scope("moe"):
        gate_params = FusedMoeParams(
            ep.gate_weight,
            perm_to_group_indices.group_indices,
            permute_indices=None,
            gather=False,
            scatter=False,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=None
        )
        gate = fused_moe(
            perm_to_group_indices.tokens, 
            gate_params,
            autotune_mode=autotune_mode
        )
        gate = ep.activation(gate)

        up_params = gate_params
        up_params.weight = ep.up_weight 
        up = fused_moe(
            perm_to_group_indices.tokens, 
            up_params,
            autotune_mode=autotune_mode
        )

        gated = gate * up

        down_params = FusedMoeParams(
            ep.down_weight,
            perm_to_group_indices.group_indices,
            None,
            gather=False,
            scatter=False,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=None
        )
        down = fused_moe(
            gated,
            down_params,
            autotune_mode=autotune_mode
        )
        
        perm_to_group_indices.tokens = down
        
    with proton.scope("output_permute"):
        output = expert_output_permute(perm_to_group_indices, topk_scores, params.topk, input.size())
    return output
