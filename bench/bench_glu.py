from functools import partial
import math
from types import NoneType
import torch
from typing import List, Callable
from triton.testing import perf_report, do_bench, Benchmark
from moe_explore.functional.topk_moe import (
    topk_moe_torch,
    topk_moe
)
from moe_explore.router import topk_router, ernie_router
from moe_explore.testing import random_interleaved_glu, random_topk_router, random_ernie_router
from moe_explore.params import MOEParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode

from moe_explore.baseline.huggingface import HAVE_HUGGINGFACE, make_huggingface_moe, set_huggingface_moe_weights
from moe_explore.baseline.scattermoe import HAVE_SCATTERMOE, make_scattermoe_mlp, scattermoe_forward
from moe_explore.baseline.transformer_engine import HAVE_TRANSFORMER_ENGINE, TransformerEngineMoE

def bench(func: Callable, quantiles: List[float], forward_backward: bool = False):
    if forward_backward:
        def inner():
            output = func()
            output.sum().backward()
            return NoneType
        func_to_call = inner
    else:
        func_to_call = func
    return do_bench(lambda: func_to_call(), quantiles=quantiles, warmup=200, rep=400)

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

def moe_benchmark(plot_name, model_name, num_experts, act_experts, hidden_dim, input_dim, activation):
    line_vals = ["fused"]
    line_names = ["Fused MoE"]
    if HAVE_HUGGINGFACE:
        line_vals.append("huggingface")
        line_names.append("Huggingface")
    if HAVE_SCATTERMOE:
        line_vals.append("scattermoe")
        line_names.append("ScatterMoE")
    if HAVE_TRANSFORMER_ENGINE:
        line_vals.append("transformer-engine")
        line_names.append("Transformer Engine")
    return Benchmark(
        x_names=["seq_len"],
        x_vals=list(range(64, 4096, 512)),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[
            ("green", "-"), 
            ("blue", "-"),
            ("black", "--"),
            ("purple", "--")
        ],
        ylabel="ms",
        plot_name=plot_name,
        args={
            "model_name": model_name,
            "num_experts": num_experts,
            "act_experts": act_experts,
            "hidden_dim": hidden_dim,
            "input_dim": input_dim,
            "activation": activation 
        })

configs = []
configs.append(
        moe_benchmark(
            "Qwen3-30B-A3B_experts=8_128",
            model_name="qwen3",
            num_experts=128,
            act_experts=8,
            input_dim=2048,
            hidden_dim=768,
            activation="swiglu"
        ))
configs.append(
        moe_benchmark(
            "OLMoE-1B-7B_experts=8_64",
            model_name="olmoe",
            num_experts=64,
            act_experts=8,
            input_dim=2048,
            hidden_dim=1024,
            activation="swiglu"
        ))
configs.append(
        moe_benchmark(
            "Ernie4.5_experts=6_64",
            model_name="ernie4",
            num_experts=64,
            act_experts=6,
            input_dim=2560,
            hidden_dim=1536,
            activation="swiglu"
        ))

@perf_report(configs)
def benchmark_moe_forward(
    model_name,
    num_experts,
    act_experts,
    hidden_dim, 
    input_dim, 
    activation, 
    seq_len, 
    provider
):
    torch.manual_seed(0)
    torch._dynamo.reset()
    
    if model_name == "ernie4":
        router = ernie_router
        router_params = random_ernie_router(
            num_experts=num_experts,
            hidden_dim=input_dim,
            topk=act_experts,
            device="cuda",
            dtype=torch.bfloat16
        )
    else:
        router = topk_router
        router_params = random_topk_router(
            num_experts=num_experts,
            hidden_dim=input_dim,
            topk=act_experts,
            softmax_before_topk=True,
            normalize_routing=True if model_name == "qwen3" else False,
            device="cuda",
            dtype=torch.bfloat16,
        )
    
    glu_params = random_interleaved_glu(
        num_experts=num_experts,
        hidden_dim=input_dim,
        intermediate_dim=hidden_dim,
        activation=activation,
        device="cuda",
        dtype=torch.bfloat16
    )
    moe_params = MOEParams(
        router_params=router_params,
        expert_params=glu_params,
        num_experts=num_experts,
        topk=act_experts
    )
    input = torch.randn((seq_len, input_dim), device=torch.device("cuda"), dtype=torch.bfloat16)

    input.requires_grad = True
    moe_params.expert_params.weight1.requires_grad = True
    moe_params.expert_params.weight2.requires_grad = True

    quantiles = [0.5, 0.2, 0.8]
    autotune_mode = AutotuneMode.FAST
    if provider == "torch":
        ms, min_ms, max_ms = bench(
            lambda: topk_moe_torch(input, moe_params, autotune_mode), 
            quantiles=quantiles, 
            forward_backward=False)
    if provider == "fused":
        ms, min_ms, max_ms = bench(
            lambda: topk_moe(input, moe_params, autotune_mode), 
            quantiles=quantiles,
            forward_backward=False)
    elif provider == "huggingface":
        moe = make_huggingface_moe(input_dim, hidden_dim, num_experts, act_experts, activation, torch.bfloat16)
        moe = moe.to(torch.bfloat16).to("cuda")
        set_huggingface_moe_weights(moe, router_params.router_weight, glu_params.weight1, glu_params.weight2)
        ms, min_ms, max_ms = bench(
            lambda: moe(input.view(1, *input.size())),
            quantiles=quantiles,
            forward_backward=False)
    elif provider == "scattermoe":
        mlp = make_scattermoe_mlp(input_dim, hidden_dim, num_experts, act_experts, activation)
        mlp = mlp.to(torch.bfloat16).to("cuda")
        ms, min_ms, max_ms = bench(
            lambda: scattermoe_forward(input, router, router_params, mlp, act_experts),
            quantiles=quantiles,
            forward_backward=False)
    elif provider == "transformer-engine":
        print(input.size(), input_dim, hidden_dim)
        moe = TransformerEngineMoE(input_dim, hidden_dim, num_experts, act_experts, activation, torch.bfloat16)
        moe = moe.to("cuda")
        router_func = lambda x: router(x, router_params)
        ms, min_ms, max_ms = bench(
            lambda: moe(input, router_func),
            quantiles=quantiles,
            forward_backward=False)
    return ms, min_ms, max_ms

benchmark_moe_forward.run(print_data=True, save_path="./")