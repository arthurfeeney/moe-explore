import math
import torch
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.testing import random_topk_router, random_interleaved_glu, random_routing, perfect_routing

def run_profile(num_tokens):
    hidden_dim = 2048
    intermediate_dim = 768
    num_experts = 128
    topk = 8

    topk_scores, topk_indices = perfect_routing(num_tokens, num_experts, topk, device="cuda", dtype=torch.bfloat16)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    num_tokens = num_tokens * topk
    input = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((num_experts, hidden_dim, intermediate_dim), device="cuda", dtype=torch.bfloat16)
    params = MGroupedGEMMParams(
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None,
        activation=None
    )

    input2 = torch.randn((num_experts, num_tokens // num_experts, hidden_dim), device="cuda", dtype=torch.bfloat16)
    weight2 = torch.randn((num_experts, hidden_dim, intermediate_dim), device="cuda", dtype=torch.bfloat16) * 0.023
    
    torch.cuda.profiler.cudart().cudaProfilerStart()
    m_grouped_gemm(input, weight, p.group_indices, params, AutotuneMode.NONE)
    torch.bmm(input2, weight2)
    torch.cuda.profiler.cudart().cudaProfilerStop()
    
run_profile(num_tokens=2048)
run_profile(num_tokens=4096)
run_profile(num_tokens=13824)