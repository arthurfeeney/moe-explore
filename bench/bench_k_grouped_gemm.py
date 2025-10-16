from functools import partial
import math
import torch
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.k_grouped_gemm import k_grouped_gemm, KGroupedGEMMParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.testing import random_groups, random_routing, random_skewed_routing, perfect_routing
from triton.testing import perf_report, do_bench, Benchmark, do_bench_cudagraph

def k_grouped_gemm_benchmark_extensive(
    plot_name, 
    routing_func,
    num_groups, 
    M,
    N, 
    topk,
    dtype
):
    line_vals = ["gemm-reference", "grouped-only", "grouped-gather-a", "grouped-gather-b", "grouped-gather-both"]
    line_names = ["GEMM-reference", "Grouped-only", "Grouped+GatherA", "Grouped+GatherB", "Grouped+GatherBoth"]
    return Benchmark(
        x_names=["num_tokens"],
        x_vals=list(range(256, 5000 + 1, 256)),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[
            ("green", "-"), 
            ("blue", "-"), 
            ("red", "-"),
            ("purple", "-"),
            ("orange", "-")
        ],
        ylabel="ms",
        plot_name=plot_name,
        args={
            "routing_func": routing_func,
            "num_groups": num_groups,
            "M": M,
            "N": N,
            "topk": topk,
            "dtype": dtype 
        })

configs = [
    k_grouped_gemm_benchmark_extensive(
        "OLMoE-1B-7B-style GEMM2, perfect routing",
        routing_func=perfect_routing,
        num_groups=64,
        M=1024,
        N=2048,
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
    return do_bench(lambda: f(), quantiles=quantiles, warmup=200, rep=400)
    #return do_bench_cudagraph(lambda: f(), quantiles=quantiles, rep=100)

dist = torch.randn
#dist = torch.zeros

def benchmark_k_grouped_gemm_only(num_tokens, num_groups, M, N, topk, dtype, p):
    a = dist((num_tokens * topk, M), device=torch.device("cuda"), dtype=dtype)
    b = dist((num_tokens * topk, N), device=torch.device("cuda"), dtype=dtype)
    params = KGroupedGEMMParams(
        permute_indices=None,
        gather_a=False,
        gather_b=False,
        num_tokens=num_tokens,
        topk=topk
    )
    return bench(lambda: k_grouped_gemm(a, b, p.group_indices, params, AutotuneMode.NONE))

def benchmark_k_grouped_gemm_gather_a(num_tokens, num_groups, M, N, topk, dtype, p):
    a = dist((num_tokens, M), device=torch.device("cuda"), dtype=dtype)
    b = dist((num_tokens * topk, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = KGroupedGEMMParams(
        permute_indices=p.indices,
        gather_a=True,
        gather_b=False,
        num_tokens=num_tokens,
        topk=topk
    )
    return bench(lambda: k_grouped_gemm(a, b, p.group_indices, params, AutotuneMode.NONE))

def benchmark_k_grouped_gemm_gather_b(num_tokens, num_groups, M, N, topk, dtype, p):
    a = dist((num_tokens * topk, M), device=torch.device("cuda"), dtype=dtype)
    b = dist((num_tokens * topk, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = KGroupedGEMMParams(
        permute_indices=p.indices,
        gather_a=False,
        gather_b=True,
        num_tokens=num_tokens,
        topk=topk
    )
    return bench(lambda: k_grouped_gemm(a, b, p.group_indices, params, AutotuneMode.NONE))

def benchmark_k_grouped_gemm_gather_both(num_tokens, num_groups, M, N, topk, dtype, p):
    a = dist((num_tokens, M), device=torch.device("cuda"), dtype=dtype)
    b = dist((num_tokens * topk, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    params = KGroupedGEMMParams(
        permute_indices=p.indices,
        gather_a=True,
        gather_b=True,
        num_tokens=num_tokens,
        topk=topk
    )
    return bench(lambda: k_grouped_gemm(a, b, p.group_indices, params, AutotuneMode.NONE))

def benchmark_gemm_reference(num_tokens, num_groups, M, N, topk, dtype):
    r"""
    This is just a normal GEMM with the same number of FLOPs to other benchmarks
    benchmarks. This is just used a reference for performance.
    """
    num_tokens = num_tokens * topk
    assert num_tokens % num_groups == 0
    input = dist((num_groups, num_tokens // num_groups, M), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, num_tokens // num_groups, N), device=torch.device("cuda"), dtype=dtype) / math.sqrt(N)
    quantiles = [0.5, 0.2, 0.8]
    f = torch.bmm
    f(input.permute(0, 2, 1), weight)
    return bench(lambda: f(input.permute(0, 2, 1), weight))

def grouped_flops(num_tokens, num_groups, M, N, topk):
    # This is really \sum_{k in group} M * N * k * 2,
    # but this factors to 2MN \sum_{k in group} k = 2MN * num_tokens * topk,
    num_tokens = num_tokens * topk
    return M * num_tokens * N * 2

@perf_report(configs)
def benchmark_k_grouped_gemm_forward(
    routing_func,
    num_tokens,
    num_groups,
    M,
    N,
    topk,
    dtype,
    provider
):
    # This benchmark uses random routers, so setting a seed for reproducible performance.
    # Ideally, results for this benchmark should be averaged over multiple runs with different seeds.
    torch.manual_seed(0)

    if provider == "gemm-reference":
        ms, _, _ = benchmark_gemm_reference(num_tokens, num_groups, M, N, topk, dtype)
    if provider in ("grouped-only", "grouped-gather-a", "grouped-gather-b", "grouped-gather-both"):
        topk_scores, topk_indices = routing_func(num_tokens, num_groups, topk, device="cuda", dtype=dtype)
        p = get_token_indices(
            topk_indices.view(-1),
            topk,
            num_groups,
            zero_prefix=True
        )
        if provider == "grouped-only":
            ms, _, _ = benchmark_k_grouped_gemm_only(num_tokens, num_groups, M, N, topk, dtype, p)
        if provider == "grouped-gather-a":
            ms, _, _ = benchmark_k_grouped_gemm_gather_a(num_tokens, num_groups, M, N, topk, dtype, p)
        if provider == "grouped-gather-b":
            ms, _, _ = benchmark_k_grouped_gemm_gather_b(num_tokens, num_groups, M, N, topk, dtype, p)
        if provider == "grouped-gather-both":
            ms, _, _ = benchmark_k_grouped_gemm_gather_both(num_tokens, num_groups, M, N, topk, dtype, p)

    flops = grouped_flops(num_tokens, num_groups, M, N, topk)
    tflops = flops / (ms / 1000) * 1e-12
    return tflops

benchmark_k_grouped_gemm_forward.run(print_data=True, save_path="./")