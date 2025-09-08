r"""
This script is used to run autotuning and determine some decent default
configuratios, so autotuning doesn't have to be run every time.
"""

import argparse
from dataclasses import dataclass
import math
import torch
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.m_grouped_glu import m_grouped_glu, MGroupedGLUParams
from moe_explore.triton_kernels.m_grouped_glu_interleaved import m_grouped_glu_interleaved, MGroupedGLUInterleavedParams
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import random_routing

import os

def autotune_grouped_gemm(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype
):
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   

    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None,
    )
    
    m_grouped_gemm(input, params, AutotuneMode.MAX)


def autotune_grouped_gemm_gather(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype
):
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   

    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        None
    )
    
    m_grouped_gemm(input, params, AutotuneMode.MAX)
    
def autotune_grouped_gemm_scatter(
    num_tokens_times_topk,
    num_experts,
    K,
    N,    
    topk,
    dtype
):
    input = torch.randn((num_tokens_times_topk, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    
    # this test is setup to implicitly assume that we have already routed `num_tokens`
    # to `num_tokens * topk`. The routing and scales are generated using the original
    # `num_tokens`, because it ends up reducing after the scales.
    num_tokens = num_tokens_times_topk // topk
    topk_scores, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        p.indices,
        False,  
        True,
        num_tokens,
        topk,
        topk_scores
    )

    m_grouped_gemm(input, params, AutotuneMode.MAX)
    
def autotune_grouped_glu_gather(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype
):
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    gate_weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    up_weight = torch.randn((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   

    params = MGroupedGLUParams(
        gate_weight,
        up_weight,
        p.group_indices,
        p.indices,
        True,
        num_tokens,
        topk,
        "silu"
    )
    
    m_grouped_glu(input, params, AutotuneMode.MAX)
    
def autotune_grouped_glu_interleaved_gather(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype
):
    input = torch.randn((num_tokens, K), dtype=dtype, device="cuda")
    weight = torch.randn((num_experts, K, 2 * N), dtype=dtype, device="cuda") / math.sqrt(N) 
    _, topk_indices = random_routing(num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   
    
    params = MGroupedGLUInterleavedParams(
        weight,
        p.group_indices,
        p.indices,
        True,
        num_tokens,
        topk,
        "silu"
    )
    
    m_grouped_glu_interleaved(input, params, AutotuneMode.MAX)
    
@dataclass
class MoESettings:
    num_experts: int
    N: int
    K: int
    topk: int

def qwen_settings(kenrel_type: str):
    if kenrel_type in ("grouped", "gather", "glu-gather", "glu-interleaved-gather"):
        N, K = 768, 2048
    elif kenrel_type == "scatter":
        N, K = 2048, 768
    return MoESettings(
        num_experts=128,
        N=N,
        K=K,
        topk=8
    )
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", 
        type=str, 
        required=True, 
        choices=["grouped", "gather", "scatter", "glu-gather", "glu-interleaved-gather"])
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["qwen"])
    args = parser.parse_args()

    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    num_tokens = args.num_tokens
    q = qwen_settings(args.kernel)
    num_experts = q.num_experts
    N = q.N
    K = q.K
    topk = q.topk
    dtype = torch.bfloat16
    
    if args.kernel == "grouped":
        autotune_grouped_gemm(num_tokens, num_experts, K, N, topk, dtype)
    elif args.kernel == "gather":
        autotune_grouped_gemm_gather(num_tokens, num_experts, K, N, topk, dtype)
    elif args.kernel == "scatter":
        autotune_grouped_gemm_scatter(num_tokens, num_experts, K, N, topk, dtype)
    elif args.kernel == "glu-gather":
        autotune_grouped_glu_gather(num_tokens, num_experts, K, N, topk, dtype)
    elif args.kernel == "glu-interleaved-gather":
        autotune_grouped_glu_interleaved_gather(num_tokens, num_experts, K, N, topk, dtype)

if __name__ == "__main__":
    main()