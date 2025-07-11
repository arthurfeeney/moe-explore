import math
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from moe_explore.functional.forward import (
    topk_moe_naive_forward,
    topk_moe_matmul_gather_scatter_forward,
    topk_moe_group_gemm_forward
)

def generate_inputs(
    device,
    dtype,
    seq_len,
    input_dim,
    hidden_dim,
    num_experts
):
    input = torch.randn((seq_len, input_dim), device=device, dtype=dtype)
    router_weight = torch.randn((input_dim, num_experts), device=device, dtype=dtype) / math.sqrt(input_dim)
    expert_weights1 = torch.randn((num_experts, input_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(input_dim)
    expert_weights2 = torch.randn((num_experts, hidden_dim, input_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim)
    return (input, router_weight, expert_weights1, expert_weights2)

seq_len = 16 * 2048
input_dim = 256
hidden_dim = 256
num_experts = 64
topk = 6
activation = torch.nn.functional.gelu
input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
        "cuda", torch.float16, seq_len, input_dim, hidden_dim, num_experts)

f = lambda: topk_moe_matmul_gather_scatter_forward(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation
    )

for i in range(5): f()

#with profile(activities=[
#        ProfilerActivity.CPU, 
#        ProfilerActivity.CUDA
#    ]) as prof:
torch.cuda.cudart().cudaProfilerStart()
f()
torch.cuda.cudart().cudaProfilerStop()

#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#prof.export_chrome_trace("trace.json")
