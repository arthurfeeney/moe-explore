from functools import partial
import math
import torch
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.testing import random_groups, random_routing, random_skewed_routing, perfect_routing
from triton.testing import perf_report, do_bench, Benchmark, do_bench_cudagraph

def m_grouped_gemm_benchmark_extensive(
    plot_name, 
    routing_func,
    num_groups, 
    N, 
    K, 
    topk, 
    dtype
):
    line_vals = ["gemm-reference", "grouped-only", "grouped-gather", "grouped-scatter"]
    line_names = ["GEMM-reference", "Grouped-only", "Grouped+Gather", "Grouped+Scatter"]
    return Benchmark(
        x_names=["num_tokens"],
        x_vals=list(range(256, 8192 + 1, 256)),
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

m_grouped_gemm_benchmark_extensive(
    "Qwen3-30B-A3B-style GEMM, balanced routing",
    routing_func=random_routing,
    num_groups=128,
    N=2048,
    K=768,
    topk=8,
    dtype=torch.bfloat16
),
m_grouped_gemm_benchmark_extensive(
    "Qwen3-30B-A3B-style GEMM, skewed routing",
    routing_func=partial(random_skewed_routing, num_skewed_experts=8, skew_factor=16),
    num_groups=128,
    N=2048,
    K=768,
    topk=8,
    dtype=torch.bfloat16
),
m_grouped_gemm_benchmark_extensive(
    "Qwen3-30B-A3B-style GEMM1, perfect routing",
    routing_func=perfect_routing,
    num_groups=128,
    N=768,
    K=2048,
    topk=8,
    dtype=torch.bfloat16
),
m_grouped_gemm_benchmark_extensive(
    "Qwen3-30B-A3B-style GEMM2, perfect routing",
    routing_func=perfect_routing,
    num_groups=128,
    N=2048,
    K=768,
    topk=8,
    dtype=torch.bfloat16
),
m_grouped_gemm_benchmark_extensive(
    "OLMoE-1B-7B-style GEMM1, perfect routing",
    routing_func=perfect_routing,
    num_groups=64,
    N=1024,
    K=2048,
    topk=8,
    dtype=torch.bfloat16
),

m_grouped_gemm_benchmark_extensive(
    "OLMoE-1B-7B-style GEMM1, perfect routing",
    routing_func=perfect_routing,
    num_groups=64,
    N=1024,
    K=2048,
    topk=8,
    dtype=torch.bfloat16
),

configs = [
    m_grouped_gemm_benchmark_extensive(
        "OLMoE-1B-7B-style GEMM2, perfect routing",
        routing_func=perfect_routing,
        num_groups=64,
        N=2048,
        K=1024,
        topk=8,
        dtype=torch.bfloat16
    ),
]

"""
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
"""

def bench(f):
    quantiles = [0.5, 0.2, 0.8]
    #return do_bench(lambda: f(), quantiles=quantiles, warmup=200, rep=400)
    return do_bench_cudagraph(lambda: f(), quantiles=quantiles, rep=100)

dist = torch.randn
#dist = torch.zeros

def benchmark_m_grouped_gemm_only(num_tokens, num_groups, N, K, topk, dtype, p):
    num_tokens = num_tokens * topk
    input = dist((num_tokens, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype)
    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None
    )
    return bench(lambda: m_grouped_gemm(input, params, AutotuneMode.NONE))

def benchmark_m_grouped_gemm_gather(num_tokens, num_groups, N, K, topk, dtype, p):
    input = dist((num_tokens, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        permute_indices=p.indices,
        gather=True,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None
    )
    return bench(lambda: m_grouped_gemm(input, params, AutotuneMode.NONE))

def benchmark_m_grouped_gemm_scatter(num_tokens, num_groups, N, K, topk, dtype, p, topk_scores):
    num_tokens_times_top = num_tokens * topk
    input = dist((num_tokens_times_top, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = MGroupedGEMMParams(
        weight,
        p.group_indices,
        permute_indices=p.indices,
        gather=False,
        scatter=True,
        num_tokens=num_tokens,
        topk=topk,
        scales=topk_scores
    )
    return bench(lambda: m_grouped_gemm(input, params, AutotuneMode.NONE))

def benchmark_gemm_reference(num_tokens, num_groups, N, K, topk, dtype):
    r"""
    This is just a normal GEMM with the same number of FLOPs to other benchmarks
    benchmarks. This is just used a reference for performance.
    """
    num_tokens = num_tokens * topk
    assert num_tokens % num_groups == 0
    input = dist((num_groups, num_tokens // num_groups, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    quantiles = [0.5, 0.2, 0.8]
    f = torch.bmm
    f(input, weight)
    return bench(lambda: f(input, weight))

def grouped_flops(num_tokens, num_groups, N, K, topk):
    num_tokens = num_tokens * topk
    return num_tokens * N * K * 2

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
    # This benchmark uses random routers, so setting a seed for reproducible performance.
    # Ideally, results for this benchmark should be averaged over multiple runs with different seeds.
    torch.manual_seed(0)

    if provider == "gemm-reference":
        ms, _, _ = benchmark_gemm_reference(num_tokens, num_groups, N, K, topk, dtype)
    if provider in ("grouped-only", "grouped-gather", "grouped-scatter"):
        topk_scores, topk_indices = routing_func(num_tokens, num_groups, topk, device="cuda", dtype=dtype)
        p = get_token_indices(
            topk_indices.view(-1),
            topk,
            num_groups,
            zero_prefix=True
        )
        if provider == "grouped-only":
            ms, _, _ = benchmark_m_grouped_gemm_only(num_tokens, num_groups, N, K, topk, dtype, p)
        if provider == "grouped-gather":
            ms, _, _ = benchmark_m_grouped_gemm_gather(num_tokens, num_groups, N, K, topk, dtype, p)
        if provider == "grouped-scatter":
            ms, _, _ = benchmark_m_grouped_gemm_scatter(num_tokens, num_groups, N, K, topk, dtype, p, topk_scores)

    flops = grouped_flops(num_tokens, num_groups, N, K, topk)
    tflops = flops / (ms / 1000) * 1e-12
    return tflops

benchmark_m_grouped_gemm_forward.run(print_data=True, save_path="./")