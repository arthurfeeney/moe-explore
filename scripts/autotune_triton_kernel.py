r"""
This script is used to run autotuning and determine some decent default
configuratios, so autotuning doesn't have to be run every time.
"""

import math
import torch
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.fused_moe import fused_moe, m_grouped_gemm_persistent_kernel, FusedMoeParams
from moe_explore.expert_permute import get_token_indices
from moe_explore.testing import random_routing

import os

def autotune_gather(
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
    
    fused_moe(input, params, AutotuneMode.FAST)
    
def autotune_scatter(
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

    fused_moe(input, params, AutotuneMode.FAST)

def main():
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    num_tokens = 32000
    num_experts = 128
    K = 2048
    N = 768
    topk = 8
    dtype = torch.bfloat16

    #autotune_gather(num_tokens, num_experts, K, N, topk, dtype)
    autotune_scatter(num_tokens, num_experts, K, N, topk, dtype)
    
if __name__ == "__main__":
    main()