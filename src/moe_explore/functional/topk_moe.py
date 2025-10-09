import torch
import triton.profiler as proton
from moe_explore.params import MOEParams, MLPParams
from moe_explore.functional.activation import activation
from moe_explore.functional.scale_and_reduce import scale_and_reduce
from moe_explore.router import router
from moe_explore.expert_permute import get_token_indices
#from moe_explore.functional.m_grouped_gemm import m_grouped_gemm
from moe_explore.functional.m_grouped_mlp import m_grouped_mlp
from moe_explore.expert_permute import expert_input_permute, expert_output_permute

@torch.compile(fullgraph=True)
def topk_moe_forward(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    topk_scores, topk_indices = router(input, params.router_params)
    perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)

    down = m_grouped_mlp(
        input,
        ep.weight1,
        ep.weight2,
        perm_to_group_indices.group_indices,
        perm_to_group_indices.indices,
        input.size(0),
        params.topk,
        ep.activation
    )

    down = scale_and_reduce(down, topk_scores, input.size(0), params.topk, down.size(-1))

    return down

def topk_moe_unfused_forward(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)

    with proton.scope("input_permute"):
        grouped_tokens = expert_input_permute(input, topk_indices, params.num_experts, params.topk)

    with proton.scope("mlp"):
        grouped_tokens.tokens = m_grouped_mlp(
            grouped_tokens.tokens,
            ep.weight1,
            ep.weight2,
            grouped_tokens.group_indices,
            None,
            input.size(0),
            params.topk,
            ep.activation
        )

    with proton.scope("output_permute"): 
        down = expert_output_permute(grouped_tokens, topk_scores, params.topk, grouped_tokens.tokens.shape)

    return down

@torch.compile
def topk_moe_torch(
    input,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)
        flat_expert_weights = topk_scores.view(-1, 1)
    with proton.scope("get_token_indices"):
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
                
                expert_up = expert_tokens @ ep.weight1[expert_id]
                expert_up = activation(expert_up, ep.activation)
                expert_out = expert_up @ ep.weight2[expert_id]
                
                # scale by scores and reduce
                expert_out.mul_(flat_expert_weights[perm_to_group_indices.indices[start_idx:end_idx]])
                
                # Autograd complains about in-place scatter_reduce_
                expert_cache = torch.scatter_reduce(
                    expert_cache,
                    0,
                    exp_token_idxs.view(-1, 1).repeat(1, input.shape[-1]),
                    expert_out,
                    reduce='sum',
                    include_self=True
                )
        return expert_cache.to(input.dtype)