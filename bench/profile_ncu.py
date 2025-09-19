import math
import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm,
    moe_glu_grouped_gemm_fused,
    moe_glu_interleaved
)
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, m_grouped_gemm_persistent_kernel, MGroupedGEMMParams
from moe_explore.params import MOEParams
from moe_explore.testing import random_glu, random_topk_router, random_interleaved_glu, random_routing, perfect_routing
import triton.profiler as proton

num_tokens = 32000
hidden_dim = 2048
intermediate_dim = 768
num_experts = 128
topk = 8
activation = "silu"

router_params = random_topk_router(num_experts, hidden_dim, topk, True, False, "cuda", torch.bfloat16)
expert_params = random_interleaved_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    "cuda",
    torch.bfloat16,
)

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
    weight,
    p.group_indices,
    permute_indices=None,
    gather=False,
    scatter=False,
    num_tokens=num_tokens,
    topk=topk,
    scales=None
)

input2 = torch.randn((num_experts, num_tokens // num_experts, hidden_dim), device="cuda", dtype=torch.bfloat16)
weight2 = torch.randn((num_experts, hidden_dim, intermediate_dim), device="cuda", dtype=torch.bfloat16)

torch.cuda.profiler.cudart().cudaProfilerStart()
m_grouped_gemm(input, params, AutotuneMode.NONE)
torch.bmm(input2, weight2)
torch.cuda.profiler.cudart().cudaProfilerStop()