import torch
import triton.profiler as proton
from moe_explore.functional.activation import activation
from moe_explore.functional.scale_and_reduce import scale_and_reduce
from moe_explore.router import router
from moe_explore.expert_permute import get_token_indices, expert_input_permute, expert_output_permute
from moe_explore.params import MOEParams, GLUParams, GLUInterleavedParams
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.m_grouped_glu import m_grouped_glu, MGroupedGLUParams
from moe_explore.triton_kernels.m_grouped_glu_interleaved import m_grouped_glu_interleaved, MGroupedGLUInterleavedParams

@torch.compile
def moe_glu_torch(
    input,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)
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
                expert_gate = activation(expert_gate, ep.activation)
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
    ep: MGroupedGLUParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)
    with proton.scope("get_token_indices"):
        perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)

    with proton.scope("fused_grouped_glu"):
        glu_params = MGroupedGLUParams(
            ep.gate_weight,
            ep.up_weight,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=True,
            num_tokens=num_tokens,
            topk=params.topk,
            activation=ep.activation
        )
        
        gated = m_grouped_glu(input, glu_params, autotune_mode=autotune_mode)
        
    with proton.scope("grouped_gemm"):
        down_params = MGroupedGEMMParams(
            ep.down_weight,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=False,
            scatter=True,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=topk_scores
        )

        down = m_grouped_gemm(
            gated,
            down_params,
            autotune_mode=autotune_mode
        )
        
    with proton.scope("scale_and_reduce"):
        down = scale_and_reduce(down, down_params.scales, down_params.num_tokens, params.topk, down.size(-1))


    return down

def moe_glu_interleaved(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: GLUInterleavedParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)
    with proton.scope("get_token_indices"):
        perm_to_group_indices = get_token_indices(topk_indices, params.topk, params.num_experts, zero_prefix=True)
            
    with proton.scope("fused_grouped_glu"):
        glu_params = MGroupedGLUInterleavedParams(
            ep.interleaved_weight,
            perm_to_group_indices.group_indices,
            perm_to_group_indices.indices,
            gather=True,
            num_tokens=input.size(0),
            topk=params.topk,
            activation=ep.activation
        )
        glu = m_grouped_glu_interleaved(input, glu_params, autotune_mode=autotune_mode)
        
    with proton.scope("down_grouped_gemm"):
        down_params = MGroupedGEMMParams(
            ep.down_weight,
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

    return down

def moe_glu_grouped_gemm(
    input: torch.Tensor,
    params: MOEParams,
    autotune_mode = None
):
    ep: MGroupedGLUParams = params.expert_params
    with proton.scope("router"):
        topk_scores, topk_indices = router(input, params.router_params)
    with proton.scope("input_permute"):
        perm_to_group_indices = expert_input_permute(input, topk_indices, num_experts=params.num_experts, topk=params.topk)
        
    num_tokens = input.size(0) * params.topk
    
    with proton.scope("gate_grouped_gemm"):
        gate_params = MGroupedGEMMParams(
            ep.gate_weight,
            perm_to_group_indices.group_indices,
            permute_indices=None,
            gather=False,
            scatter=False,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=None
        )
        gate = m_grouped_gemm(
            perm_to_group_indices.tokens, 
            gate_params,
            autotune_mode=autotune_mode
        )
    with proton.scope("activation"):
        gate = activation(gate, ep.activation)

    with proton.scope("up_grouped_gemm"):
        up_params = gate_params
        up_params.weight = ep.up_weight 
        up = m_grouped_gemm(
            perm_to_group_indices.tokens, 
            up_params,
            autotune_mode=autotune_mode
        )

    with proton.scope("gating"):
        gated = gate * up

    with proton.scope("down_grouped_gemm"):
        down_params = MGroupedGEMMParams(
            ep.down_weight,
            perm_to_group_indices.group_indices,
            None,
            gather=False,
            scatter=False,
            num_tokens=num_tokens,
            topk=params.topk,
            scales=None
        )
        down = m_grouped_gemm(
            gated,
            down_params,
            autotune_mode=autotune_mode
        )
        
        perm_to_group_indices.tokens = down
        
    with proton.scope("output_permute"):
        output = expert_output_permute(perm_to_group_indices, topk_scores, params.topk, input.size())
    return output
