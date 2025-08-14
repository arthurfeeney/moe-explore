from functools import partial
import math
import torch
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.fused_moe import fused_moe, FusedMoeParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.testing import random_groups, random_routing, random_skewed_routing, random_olmoe_routing
from triton.testing import perf_report, do_bench, Benchmark

def m_grouped_gemm_benchmark(plot_name, routing_func, num_groups, N, K, topk, dtype):
    line_vals = ["gemm-reference", "grouped-only", "grouped-gather", "grouped-scatter"]
    line_names = ["GEMM-reference", "Grouped-only", "Grouped+Gather", "Grouped+Scatter"]
    return Benchmark(
        x_names=["num_tokens"],
        x_vals=[512, 1024, 2048, 4096, 32000],
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[
            ("green", "-"), 
            ("blue", "-"), 
            ("red", "-"),
            ("purple", "-"),
        ],
        ylabel="ms",
        plot_name=plot_name,
        args={
            "routing_func": routing_func,
            "num_groups": num_groups,
            "N": N,
            "K": K,
            "topk": topk,
            "dtype": dtype 
        })

configs = []
configs.append(
        m_grouped_gemm_benchmark(
            "Qwen3-30B-A3B-style GEMM, balanced routing",
            routing_func=random_routing,
            num_groups=128,
            N=2048,
            K=768,
            topk=8,
            dtype=torch.bfloat16
        ))
configs.append(
        m_grouped_gemm_benchmark(
            "Qwen3-30B-A3B-style GEMM, skewed routing",
            routing_func=partial(random_skewed_routing, num_skewed_experts=8, skew_factor=16),
            num_groups=128,
            N=2048,
            K=768,
            topk=8,
            dtype=torch.bfloat16
        ))
configs.append(
        m_grouped_gemm_benchmark(
            "OLMoE-1B-7B-style GEMM, balanced routing",
            routing_func=random_routing,
            num_groups=64,
            N=2048,
            K=1024,
            topk=8,
            dtype=torch.bfloat16
        ))
configs.append(
        m_grouped_gemm_benchmark(
            "OLMoE-1B-7B-style GEMM, skewed routing",
            routing_func=random_olmoe_routing, #partial(random_skewed_routing, num_skewed_experts=4, skew_factor=16),
            num_groups=64,
            N=2048,
            K=1024,
            topk=8,
            dtype=torch.bfloat16
        ))

def benchmark_m_grouped_gemm_only(num_tokens, num_groups, N, K, topk, dtype):
    num_tokens = num_tokens * topk
    input = torch.randn((num_tokens, K), device=torch.device("cuda"), dtype=dtype)
    group_indices = random_groups(num_tokens, num_groups, device=torch.device("cuda"))
    weight = torch.randn((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = FusedMoeParams(
        weight,
        group_indices,
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None
    )
    quantiles = [0.5, 0.2, 0.8]
    fused_moe(input, params, AutotuneMode.NONE)
    return do_bench(lambda: fused_moe(input, params, AutotuneMode.NONE), quantiles=quantiles)

def benchmark_m_grouped_gemm_gather(num_tokens, num_groups, N, K, topk, dtype, p):
    input = torch.randn((num_tokens, K), device=torch.device("cuda"), dtype=dtype)
    weight = torch.randn((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = FusedMoeParams(
        weight,
        p.group_indices,
        permute_indices=p.indices,
        gather=True,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None
    )
    quantiles = [0.5, 0.2, 0.8]
    fused_moe(input, params, AutotuneMode.NONE)
    return do_bench(lambda: fused_moe(input, params, AutotuneMode.NONE), quantiles=quantiles)

def benchmark_m_grouped_gemm_scatter(num_tokens, num_groups, N, K, topk, dtype, p, topk_scores):
    num_tokens_times_top = num_tokens * topk
    input = torch.randn((num_tokens_times_top, K), device=torch.device("cuda"), dtype=dtype)
    weight = torch.randn((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = FusedMoeParams(
        weight,
        p.group_indices,
        permute_indices=p.indices,
        gather=False,
        scatter=True,
        num_tokens=num_tokens,
        topk=topk,
        scales=topk_scores
    )
    quantiles = [0.5, 0.2, 0.8]
    fused_moe(input, params, AutotuneMode.NONE)
    return do_bench(lambda: fused_moe(input, params, AutotuneMode.NONE), quantiles=quantiles)

def benchmark_gemm_reference(num_tokens, num_groups, N, K, topk, dtype):
    r"""
    This is just a normal GEMM with a similar number of FLOPs to the grouped gemm
    benchmarks. This is just used a reference for performance.
    """
    num_tokens = num_tokens * topk
    assert num_tokens % num_groups == 0
    input = torch.randn((num_groups, num_tokens // num_groups, K), device=torch.device("cuda"), dtype=dtype)
    weight = torch.randn((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    quantiles = [0.5, 0.2, 0.8]
    f = torch.compile(torch.bmm)
    #f = torch.bmm
    return do_bench(lambda: f(input, weight), quantiles=quantiles)

@perf_report(configs)
def benchmark_m_grouped_gemm_forward(
    routing_func,
    num_tokens,
    num_groups,
    N,
    K,
    topk,
    dtype,
    provider
):
    if provider == "gemm-reference":
        return benchmark_gemm_reference(num_tokens, num_groups, N, K, topk, dtype)
    if provider == "grouped-only":
        return benchmark_m_grouped_gemm_only(num_tokens, num_groups, N, K, topk, dtype)
    if provider in ("grouped-gather", "grouped-scatter"):
        topk_scores, topk_indices = routing_func(num_tokens, num_groups, topk, device="cuda", dtype=dtype)
        p = get_token_indices(
            topk_indices.view(-1),
            topk,
            num_groups,
            zero_prefix=True
        )
        if provider == "grouped-gather":
            return benchmark_m_grouped_gemm_gather(num_tokens, num_groups, N, K, topk, dtype, p)
        if provider == "grouped-scatter":
            return benchmark_m_grouped_gemm_scatter(num_tokens, num_groups, N, K, topk, dtype, p, topk_scores)
    raise ValueError(f"Invalid provider: {provider}")

# This benchmark uses random routers, so setting a seed for reproducible performance.
# Ideally, results for this benchmark should be averaged over multiple runs with different seeds.
torch.manual_seed(0)
benchmark_m_grouped_gemm_forward.run(print_data=True, save_path="./")