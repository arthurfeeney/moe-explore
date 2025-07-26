from functools import partial
import math
import torch
from triton.testing import perf_report, do_bench, Benchmark
from moe_explore.functional.mlp import (
        moe_mlp_torch,
        moe_mlp_grouped_gemm_fused,
        moe_mlp_grouped_gemm
)

import sys
sys.path.append("./")
sys.path.append("bench/")
sys.path.append("bench/external")

from scattermoe.mlp import MLP as ScatterMoEMLP
from external.scattermoe import scattermoe_forward
from external.sglang_fused_moe import sglang_fused_moe

def generate_inputs(
    device,
    dtype,
    batch_size,
    seq_len,
    input_dim,
    hidden_dim,
    num_experts
):
    input = torch.randn((batch_size * seq_len, input_dim), device=device, dtype=dtype)
    router_weight = torch.randn((input_dim, num_experts), device=device, dtype=dtype) / math.sqrt(input_dim)
    expert_weights1 = torch.randn((num_experts, input_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(input_dim)
    expert_weights2 = torch.randn((num_experts, hidden_dim, input_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim)
    return (input, router_weight, expert_weights1, expert_weights2)

def moe_benchmark(plot_name, num_experts, act_experts, seq_len, hidden_dim, input_dim, activation):
    return Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 4],
        line_arg="provider",
        line_vals=["naive", "group_gemm", "fused", "scattermoe", "sglang"],
        line_names=["Naive", "Group GEMM", "Fused", "ScatterMoE", "SGLang"],
        styles=[
            ("green", "-"), 
            ("blue", "-"), 
            ("red", "-"), 
            ("orange", "--"),
            ("black", "--")
        ],
        ylabel="ms",
        plot_name=plot_name,
        args={
            "num_experts": num_experts,
            "act_experts": act_experts,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "input_dim": input_dim,
            "activation": activation 
        })

configs = []
# TODO: Need SwiGLU variant...
# TODO: this is too large for A30... damn lol
"""
configs.append(
        moe_benchmark(
            "Qwen3-30B-A3B, seq_len=64, 8/128 experts",
            num_experts=128,
            act_experts=8,
            seq_len=64,
            hidden_dim=768,
            input_dim=1024,
            activation=torch.nn.functional.gelu
        ))
"""
configs.append(
        moe_benchmark(
            "OLMoE-1B-7B, seq_len=64, 8/64 experts",
            num_experts=64,
            act_experts=8,
            seq_len=64,
            input_dim=512,
            hidden_dim=512,
            activation=torch.nn.functional.gelu
        ))

def bench_moe_explore(
    func,
    input,
    router_weight,
    expert_weights1,
    expert_weights2,
    input_dim,
    num_experts,
    act_experts,
    activation
):
    with torch.no_grad():
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = do_bench(lambda: func(
            input,
            router_weight,
            expert_weights1,
            expert_weights2,
            input_dim,
            num_experts,
            act_experts,
            activation), quantiles=quantiles)
        return ms, min_ms, max_ms

@perf_report(configs)
def benchmark_moe_forward(
    num_experts,
    act_experts,
    seq_len, 
    hidden_dim, 
    input_dim, 
    activation, 
    batch_size, 
    provider
):
    input, router_weight, expert_weights1, expert_weights2 = generate_inputs(
            "cuda", torch.float16, batch_size, seq_len, input_dim, hidden_dim, num_experts)
    
    moe_explore_options = ("naive", "group_gemm", "fused")
    if provider in moe_explore_options: 
        if provider == "naive":
            func = moe_mlp_torch
        elif provider == "group_gemm":
            func = moe_mlp_grouped_gemm
        elif provider == "fused":
            func = moe_mlp_grouped_gemm_fused
        ms, min_ms, max_ms = bench_moe_explore(
            func, 
            input,
            router_weight,
            expert_weights1,
            expert_weights2,
            input_dim,
            num_experts,
            act_experts,
            activation
        )

    elif provider == "scattermoe":
        # scatter moe is using a different set of randomly initialized weights,
        # but this should be computing effectively the same thing as moe explore
        mlp = ScatterMoEMLP(input_dim, hidden_dim, num_experts, act_experts, activation)
        mlp = mlp.to(torch.float16).to("cuda")
        ms, min_ms, max_ms = do_bench(
                lambda: scattermoe_forward(input, router_weight, mlp, act_experts),
                quantiles=[0.5, 0.2, 0.8])

    elif provider == "sglang":
        quantiles=[0.5, 0.2, 0.8]
        ms, min_ms, max_ms = do_bench(lambda: sglang_fused_moe(
            input,
            router_weight,
            expert_weights1,
            expert_weights2,
            act_experts,
            activation="gelu"
        ), quantiles=quantiles)

    else:
        print(f"provider {provider} invalid")

    return ms, min_ms, max_ms

benchmark_moe_forward.run(print_data=True)
