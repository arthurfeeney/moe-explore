import math
import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm,
    moe_glu_grouped_gemm_fused
)
from moe_explore.params import MOEParams, random_glu
import triton.profiler as proton

seq_len = 16 * 2048
input_dim = 1024
hidden_dim = 768
num_experts = 128
topk = 8
activation = torch.nn.functional.silu

expert_params = random_glu(
    num_experts,
    input_dim,
    hidden_dim,
    activation,
    "cuda",
    torch.float16,
)
moe_params = MOEParams(expert_params, num_experts=num_experts, topk=topk)

input = torch.randn((seq_len, input_dim), device="cuda", dtype=torch.float16)

def f():
    return moe_glu_grouped_gemm_fused(
        input,
        moe_params
    )

#with profile(
#    activities=[
#        ProfilerActivity.CPU, 
#        ProfilerActivity.CUDA
#    ],
#    schedule=schedule(wait=1, warmup=5, active=10),
#    with_stack=True
#) as prof:
#    for i in range(20):
#        f()
#        prof.step()

#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#print(prof)
#prof.export_chrome_trace("trace.json")

#for i in range(10): f()
#torch.cuda.cudart().cudaProfilerStart()
#f()
#torch.cuda.cudart().cudaProfilerStop()

session_id = proton.start(name="moe", context="shadow")
proton.deactivate()
# warmup
for i in range(3):
    f()
proton.activate()
f()
proton.finalize()
