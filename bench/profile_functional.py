import math
import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm,
    moe_glu_grouped_gemm_fused
)
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.triton_kernels.fused_moe import fused_moe, m_grouped_gemm_persistent_kernel
from moe_explore.params import MOEParams
from moe_explore.testing import random_glu
import triton.profiler as proton

seq_len = 32000
hidden_dim = 2048
intermediate_dim = 768
num_experts = 128
topk = 8
activation = torch.nn.functional.silu

expert_params = random_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    "cuda",
    torch.bfloat16,
)
moe_params = MOEParams(expert_params, num_experts=num_experts, topk=topk)
input = torch.randn((seq_len, hidden_dim), device="cuda", dtype=torch.bfloat16)

def f():
    return moe_glu_grouped_gemm_fused(
        input,
        moe_params,
        AutotuneMode.NONE
    )

# Do autotuning
for i in range(10): f()

torch.cuda.cudart().cudaProfilerStart()
f()
# This is used a performance comparison.
input = torch.randn((num_experts, (seq_len * topk) // num_experts, hidden_dim), device=torch.device("cuda"), dtype=torch.bfloat16)
weight = torch.randn((num_experts, hidden_dim, intermediate_dim), device=torch.device("cuda"), dtype=torch.bfloat16) / math.sqrt(intermediate_dim)
torch.bmm(input, weight)
torch.cuda.cudart().cudaProfilerStop()

#session_id = proton.start(name="moe", context="shadow")
#proton.deactivate()
## warmup
#for i in range(3):
#    f()
#proton.activate()
#f()
#proton.finalize()
