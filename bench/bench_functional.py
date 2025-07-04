import math
import torch
from triton.testing import perf_report, do_bench, Benchmark
from moe_explore.functional import (
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

num_experts = 64
seq_len = 2048
hidden_dim = 256
input_dim = 256
activation = torch.nn.functional.gelu

configs = []
configs.append(
    Benchmark(
        x_names=["topk"],
        x_vals=[2, 4, 8, 16, 32, 60],
        line_arg="provider",
        line_vals=["naive", "group_gemm", "fused"],
        line_names=["Naive", "Group GEMM", "Fused"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="MoE Performance",
        args={
            "num_experts": num_experts,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "input_dim": input_dim
        }
    )
)

@perf_report(configs)
def benchmark(num_experts, seq_len, hidden_dim, input_dim, topk, provider):
    input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
            "cuda", torch.float16, seq_len, input_dim, hidden_dim, num_experts)

    if provider == "naive":
        func = topk_moe_naive_forward
    elif provider == "group_gemm":
        func = topk_moe_group_gemm_forward
    elif provider == "fused":
        func = topk_moe_matmul_gather_scatter_forward

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = do_bench(lambda: func(
        input,
        router_weight,
        expert_weights1,
        expert_weights2,
        input_dim,
        num_experts,
        topk,
        activation), quantiles=quantiles)

    return ms, min_ms, max_ms

benchmark.run(print_data=True, save_path="./")
