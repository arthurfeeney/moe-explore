r"""
This script is used to run autotuning and determine some decent default
configuratios, so autotuning doesn't have to be run every time.
"""

import argparse
import math
import torch
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, FusedMoeParams
from moe_explore.triton_kernels.m_grouped_glu import m_grouped_glu, MGroupedGLUParams
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import random_routing

import os

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

    params = FusedMoeParams(
        weight,
        p.group_indices,
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        None
    )
    
    m_grouped_gemm(input, params, AutotuneMode.FAST)
    
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

    params = FusedMoeParams(
        weight,
        p.group_indices,
        p.indices,
        False,  
        True,
        num_tokens,
        topk,
        topk_scores
    )

    m_grouped_gemm(input, params, AutotuneMode.FAST)
    
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
    
    m_grouped_glu(input, params, AutotuneMode.FAST)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", 
        type=str, 
        required=True, 
        choices=["gather", "scatter", "glu-gather"])
    parser.add_argument("--num-tokens", type=int, required=True)
    args = parser.parse_args()

    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    num_tokens = args.num_tokens
    num_experts = 128
    K = 2048
    N = 768
    topk = 8
    dtype = torch.bfloat16
    
    if args.kernel == "gather":
        autotune_grouped_gemm_gather(num_tokens, num_experts, K, N, topk, dtype)
    elif args.kernel == "scatter":
        autotune_grouped_gemm_scatter(num_tokens, num_experts, K, N, topk, dtype)
    elif args.kernel == "glu-gather":
        autotune_grouped_glu_gather(num_tokens, num_experts, K, N, topk, dtype)
    

if __name__ == "__main__":
    main()