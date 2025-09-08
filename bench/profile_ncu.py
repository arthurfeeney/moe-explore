import math
import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm,
    moe_glu_grouped_gemm_fused,
    moe_glu_interleaved
)
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, m_grouped_gemm_persistent_kernel
from moe_explore.params import MOEParams
from moe_explore.testing import random_glu, random_topk_router, random_interleaved_glu
import triton.profiler as proton

seq_len = 2048
hidden_dim = 2048
intermediate_dim = 768
num_experts = 128
topk = 8
activation = "silu"

router_params = random_topk_router(num_experts, hidden_dim, topk, True, False, "cuda", torch.bfloat16)
expert_params = random_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    "cuda",
    torch.bfloat16,
)

moe_params = MOEParams(router_params, expert_params, num_experts=num_experts, topk=topk)
input = torch.empty((seq_len, hidden_dim), device="cuda", dtype=torch.bfloat16)

#input = torch.randn((num_experts, (seq_len * topk) // num_experts, hidden_dim), device=torch.device("cuda"), dtype=torch.bfloat16)
weight = torch.empty((num_experts, hidden_dim, intermediate_dim), device=torch.device("cuda"), dtype=torch.bfloat16) / math.sqrt(intermediate_dim)

def f1(autotune_mode):
    return moe_glu_grouped_gemm(
        input,
        moe_params,
        autotune_mode
    )

def f2(autotune_mode):
    return moe_glu_grouped_gemm_fused(
        input,
        moe_params,
        autotune_mode
    )

#f2(AutotuneMode.FAST)
#torch.cuda.profiler.cudart().cudaProfilerStart()
f2(AutotuneMode.NONE)
#torch.cuda.profiler.cudart().cudaProfilerStop()