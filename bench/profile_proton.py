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
from moe_explore.params import MOEParams
from moe_explore.testing import random_glu, random_topk_router, random_interleaved_glu
import triton.profiler as proton
import triton

seq_len = 32000
hidden_dim = 2048
intermediate_dim = 1024
num_experts = 64
topk = 8
activation = "silu"

def glu_inputs():
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
    input = torch.randn((seq_len, hidden_dim), device="cuda", dtype=torch.bfloat16)
    return moe_params, input

def glu_interleaved_inputs():
    router_params = random_topk_router(num_experts, hidden_dim, topk, True, False, "cuda", torch.bfloat16)
    expert_params = random_interleaved_glu(
        num_experts,
        hidden_dim,
        intermediate_dim,
        activation,
        "cuda",
        torch.bfloat16,
    )
    moe_params = MOEParams(router_params, expert_params, num_experts=num_experts, topk=topk)
    input = torch.randn((seq_len, hidden_dim), device="cuda", dtype=torch.bfloat16)
    return moe_params, input

session_id = proton.start(name="moe", context="shadow")

# warmup and autotuning
proton.deactivate()

# warmup gpu
moe_params, input = glu_inputs()
for i in range(2):
    moe_glu_grouped_gemm(
        input,
        moe_params,
        AutotuneMode.NONE
    )

# Clear cache after warmup and in between iterations
cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
triton.runtime.driver.active.clear_cache(cache)

moe_params, input = glu_inputs()
triton.runtime.driver.active.clear_cache(cache)
for i in range(3):
    proton.activate()
    with proton.scope("unfused"):
        moe_glu_grouped_gemm(
            input,
            moe_params,
            AutotuneMode.NONE
        )
    proton.deactivate()
    triton.runtime.driver.active.clear_cache(cache)
    
moe_params, input = glu_interleaved_inputs()
triton.runtime.driver.active.clear_cache(cache)
for i in range(3):
    proton.activate()
    with proton.scope("fused-interleaved"):
        moe_glu_interleaved(
            input,
            moe_params,
            AutotuneMode.NONE
        )
    proton.deactivate()
    triton.runtime.driver.active.clear_cache(cache)
    
proton.finalize()