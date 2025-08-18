from functools import partial
import math
import torch
from triton.testing import perf_report, do_bench, Benchmark
from moe_explore.functional.glu import (
        moe_glu_torch,
        moe_glu_grouped_gemm_fused,
        moe_glu_grouped_gemm
)
from moe_explore.testing import random_glu
from moe_explore.params import MOEParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode

import sys
sys.path.append("./")
sys.path.append("bench/")
sys.path.append("bench/external")

try:
    from scattermoe.mlp import GLUMLP as ScatterMoEGLU
    from external.scattermoe import scattermoe_forward
    HAVE_SCATTERMOE = True
except ImportError:
    HAVE_SCATTERMOE = False
    
#from external.sglang_fused_moe import sglang_fused_moe
#except ImportError:
#    print("ScatterMoE and SGLang not installed, skipping")
#    pass

def glu_tflops(num_tokens, num_experts, input_dim, hidden_dim, act_experts, ms):
    r""" This computes the flops of a GLU forward pass. Flops are counted separately,
    so an FMA is counted as two flops.
    """
    router_flop_count = 2 * num_tokens * num_experts * input_dim
    num_routed_tokens = num_tokens * act_experts
    gate_flop_count = 2 * num_routed_tokens * input_dim * hidden_dim
    up_flop_count = 2 * num_routed_tokens * input_dim * hidden_dim
    # Lower bound on FLOPs for activation
    act_flop_count = num_routed_tokens * hidden_dim
    down_flop_count = 2 * num_routed_tokens * hidden_dim * input_dim
    flop_count = router_flop_count + gate_flop_count + up_flop_count + act_flop_count + down_flop_count
    tera_flop_count = flop_count * 1e-12 
    flop_per_sec = tera_flop_count / (ms / 1000)
    return flop_per_sec

def moe_benchmark(plot_name, num_experts, act_experts, hidden_dim, input_dim, activation):
    line_vals = ["torch", "grouped_gemm", "fused"]
    line_names = ["Torch", "Grouped GLU", "Fused GLU"]
    if HAVE_SCATTERMOE:
        line_vals.append("scattermoe")
        line_names.append("ScatterMoE")
    return Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
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
            "hidden_dim": hidden_dim,
            "input_dim": input_dim,
            "activation": activation 
        })

configs = []
configs.append(
        moe_benchmark(
            "Qwen3-30B-A3B_seq_len=64_experts=8_128",
            num_experts=128,
            act_experts=8,
            input_dim=2048,
            hidden_dim=768,
            activation="silu"
        ))
configs.append(
        moe_benchmark(
            "OLMoE-1B-7B_seq_len=64_experts=8_64",
            num_experts=64,
            act_experts=8,
            input_dim=2048,
            hidden_dim=1024,
            activation="silu"
        ))

@perf_report(configs)
def benchmark_moe_forward(
    num_experts,
    act_experts,
    hidden_dim, 
    input_dim, 
    activation, 
    seq_len, 
    provider
):
    glu_params = random_glu(
        num_experts=num_experts,
        hidden_dim=input_dim,
        intermediate_dim=hidden_dim,
        activation=activation,
        device="cuda",
        dtype=torch.bfloat16
    )
    moe_params = MOEParams(
        expert_params=glu_params,
        num_experts=num_experts,
        topk=act_experts
    )
    input = torch.randn((seq_len, input_dim), device=torch.device("cuda"), dtype=torch.bfloat16)

    quantiles = [0.5, 0.2, 0.8]
    autotune_mode = AutotuneMode.NONE
    if provider == "torch":
        ms, min_ms, max_ms = do_bench(lambda: moe_glu_torch(input, moe_params, autotune_mode), quantiles=quantiles)
    elif provider == "grouped_gemm":
        ms, min_ms, max_ms = do_bench(lambda: moe_glu_grouped_gemm(input, moe_params, autotune_mode), quantiles=quantiles)
    elif provider == "fused":
        ms, min_ms, max_ms = do_bench(lambda: moe_glu_grouped_gemm_fused(input, moe_params, autotune_mode), quantiles=quantiles)
    elif provider == "scattermoe":
        glu = ScatterMoEGLU(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_experts=moe_params.num_experts,
            top_k=moe_params.topk).to("cuda").to(torch.bfloat16)
        ms, min_ms, max_ms = do_bench(
            lambda: scattermoe_forward(input, moe_params.expert_params.router_weight, glu, moe_params.topk),
            quantiles=quantiles)

    return ms, min_ms, max_ms

benchmark_moe_forward.run(print_data=True, save_path="./")