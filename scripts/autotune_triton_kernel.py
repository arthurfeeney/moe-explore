r"""
This script is used to run autotuning and determine some decent default
configuratios, so autotuning doesn't have to be run every time.
"""

import argparse
from dataclasses import dataclass
import math
import os
import torch
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.k_grouped_gemm import k_grouped_gemm, KGroupedGEMMParams
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import random_routing, random_skewed_routing, perfect_routing

def router(router_name: str, *args, **kwargs):
    if router_name == "random":
        return random_routing(*args, **kwargs)
    #elif router_name == "skewed":
    #    return skewed_routing(*args, **kwargs)
    elif router_name == "perfect":
        return perfect_routing(*args, **kwargs)
    else:
        raise ValueError(f"Invalid router name: {router_name}")

def autotune_grouped_gemm(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype,
    dist,
    router_name
):
    input = dist((num_tokens, K), dtype=dtype, device="cuda")
    weight = dist((num_experts, K, N), dtype=dtype, device="cuda")
    _, topk_indices = router(router_name, num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   

    params = MGroupedGEMMParams(
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens * topk,
        topk=1,
        scales=None,
    )
    
    m_grouped_gemm(input, weight, p.group_indices, params, AutotuneMode.MAX)

def autotune_k_grouped_gemm(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype,
    dist,
    router_name
):
    free, total = torch.cuda.memory.mem_get_info()
    print((total - free) / (1024 ** 3), free, total)
    
    # This computes lhs.T @ rhs, so the K-dimension is really the M-dimension
    # and the reduction is over the groups in the num_tokens dimension
    lhs = dist((num_tokens, K), dtype=dtype, device="cuda")
    rhs = dist((num_tokens, N), dtype=dtype, device="cuda")
    _, topk_indices = router(router_name, num_tokens, num_experts, 1, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        1,
        num_experts,
        zero_prefix=True
    )   

    params = KGroupedGEMMParams(
        permute_indices=p.indices,
        gather_a=True,
        gather_b=False,
        num_tokens=num_tokens,
        topk=1,
    )
    
    free, total = torch.cuda.memory.mem_get_info()
    print((total - free) / (1024 ** 3), free, total)
    
    k_grouped_gemm(lhs, rhs, p.group_indices, params, AutotuneMode.MAX)
    free, total = torch.cuda.memory.mem_get_info()
    print((total - free) / (1024 ** 3), free, total)


def autotune_grouped_gemm_gather(
    num_tokens,
    num_experts,
    K,
    N,
    topk,
    dtype,
    dist,
    router_name
):
    input = dist((num_tokens, K), dtype=dtype, device="cuda")
    weight = dist((num_experts, K, N), dtype=dtype, device="cuda") / math.sqrt(N) 
    _, topk_indices = router(router_name, num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )   

    params = MGroupedGEMMParams(
        p.indices,
        True,
        False,
        num_tokens,
        topk,
        None
    )
    
    m_grouped_gemm(input, weight, p.group_indices, params, AutotuneMode.MAX)
    
def autotune_grouped_gemm_scatter(
    num_tokens_times_topk,
    num_experts,
    K,
    N,    
    topk,
    dtype,
    dist,
    router_name
):
    input = dist((num_tokens_times_topk, K), dtype=dtype, device="cuda")
    weight = dist((num_experts, K, N), dtype=dtype, device="cuda")
    
    # this test is setup to implicitly assume that we have already routed `num_tokens`
    # to `num_tokens * topk`. The routing and scales are generated using the original
    # `num_tokens`, because it ends up reducing after the scales.
    num_tokens = num_tokens_times_topk // topk
    topk_scores, topk_indices = router(router_name, num_tokens, num_experts, topk, device="cuda", dtype=dtype)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    params = MGroupedGEMMParams(
        p.indices,
        False,  
        True,
        num_tokens,
        topk,
        topk_scores
    )

    m_grouped_gemm(input, weight, p.group_indices, params, AutotuneMode.MAX)

@dataclass
class MoESettings:
    num_experts: int
    N: int
    K: int
    topk: int

def qwen_settings(kenrel_type: str):
    if kenrel_type in ("grouped", "gather", "glu-gather", "glu-interleaved-gather", "k-grouped"):
        N, K = 2048, 1536
    elif kenrel_type == "scatter":
        N, K = 2048, 768
    return MoESettings(
        num_experts=128,
        N=N,
        K=K,
        topk=8
    )
    
def olmoe_settings(kenrel_type: str):
    if kenrel_type in ("grouped", "gather", "glu-gather", "glu-interleaved-gather", "k-grouped"):
        N, K = 256, 256 #2048, 2048
    elif kenrel_type == "scatter":
        N, K = 2048, 1024
    return MoESettings(
        num_experts=64,
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
        choices=["grouped", "gather", "scatter", "k-grouped"])
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "olmoe"])
    parser.add_argument(
        "--init-dist", 
        type=str, 
        required=True,
        choices=["randn", "zeros"])
    parser.add_argument("--routing", type=str, required=True, choices=["random", "skewed", "perfect"])
    args = parser.parse_args()
    
    print(args)

    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    torch.manual_seed(0)

    num_tokens = args.num_tokens
    if args.model == "qwen":
        q = qwen_settings(args.kernel)
    elif args.model == "olmoe":
        q = olmoe_settings(args.kernel)
    num_experts = q.num_experts
    N = q.N
    K = q.K
    topk = q.topk
    dtype = torch.bfloat16
    init_dist = torch.randn if args.init_dist == "randn" else torch.zeros
    
    if args.kernel == "grouped":
        autotune_grouped_gemm(num_tokens, num_experts, K, N, topk, dtype, init_dist, args.routing)
    elif args.kernel == "gather":
        autotune_grouped_gemm_gather(num_tokens, num_experts, K, N, topk, dtype, init_dist, args.routing)
    elif args.kernel == "scatter":
        autotune_grouped_gemm_scatter(num_tokens, num_experts, K, N, topk, dtype, init_dist, args.routing)
    elif args.kernel == "k-grouped":
        autotune_k_grouped_gemm(num_tokens, num_experts, K, N, topk, dtype, init_dist, args.routing)

if __name__ == "__main__":
    main()