import torch
import triton.profiler as proton
from moe_explore.params import MOEParams, MLPParams
from moe_explore.functional.activation import activation
from moe_explore.functional.scale_and_reduce import scale_and_reduce
from moe_explore.router import router
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams

def topk_moe_forward(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: MLPParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)
    with proton.scope("get_token_indices"):
        perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)
        
    with proton.scope("fused_grouped_glu"):
        glu_params = MGroupedGEMMParams(
            ep.weight1,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=True,
            scatter=False,
            num_tokens=input.size(0),
            topk=params.topk,
            scales=None,
            activation=ep.activation
        )
        glu = m_grouped_gemm(input, glu_params, autotune_mode=autotune_mode)
        
    with proton.scope("down_grouped_gemm"):
        down_params = MGroupedGEMMParams(
            ep.weight2,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=False,
            scatter=True,
            num_tokens=input.size(0),
            topk=params.topk,
            scales=topk_scores
        )

        down = m_grouped_gemm(
            glu,
            down_params,
            autotune_mode=autotune_mode
        )
        
    with proton.scope("scale_and_reduce"):
        down = scale_and_reduce(down, down_params.scales, down_params.num_tokens, params.topk, down.size(-1))

    if params.shared_expert_params is not None:
        with proton.scope("shared_expert"):
            h = input @ params.shared_expert_params.up_weight
            h = activation(h, params.shared_expert_params.activation)
            h = h @ params.shared_expert_params.down_weight
            h = h.sum(0)
            down = down + h

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
                expert_cache.scatter_reduce_(
                    0,
                    exp_token_idxs.view(-1, 1).repeat(1, input.shape[-1]),
                    expert_out,
                    reduce='sum'
                )
        return expert_cache