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

def moe_benchmark(plot_name, num_experts, seq_len, hidden_dim, input_dim, activation):
    return Benchmark(
        x_names=["topk"],
        x_vals=[2, 4, 8, 16],
        line_arg="provider",
        line_vals=["naive", "group_gemm", "fused"],
        line_names=["Naive", "Group GEMM", "Fused"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="ms",
        plot_name=plot_name,
        args={
            "num_experts": num_experts,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "input_dim": input_dim,
            "activation": activation 
        })

configs = []
configs.append(
        moe_benchmark(
            "Small Input MoE Forward",
            num_experts=64,
            seq_len=2048,
            hidden_dim=256,
            input_dim=256,
            activation=torch.nn.functional.gelu
        ))
configs.append(
        moe_benchmark(
            "Large Input MoE Forward",
            num_experts=64,
            seq_len=16 * 2048,
            hidden_dim=512,
            input_dim=512,
            activation=torch.nn.functional.gelu
        ))


@perf_report(configs)
def benchmark_moe_forward(num_experts, seq_len, hidden_dim, input_dim, activation, topk, provider):
    input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
            "cuda", torch.float16, seq_len, input_dim, hidden_dim, num_experts)

    if provider == "naive":
        func = topk_moe_naive_forward
    elif provider == "group_gemm":
        func = topk_moe_group_gemm_forward
    elif provider == "fused":
        func = topk_moe_matmul_gather_scatter_forward

    with torch.no_grad():
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

benchmark_moe_forward.run(print_data=True)
