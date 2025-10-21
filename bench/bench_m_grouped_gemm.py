from functools import partial
import math
from sympy.utilities.misc import func_name
import torch
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode
from moe_explore.testing import random_groups, random_routing, random_skewed_routing, perfect_routing
from moe_explore.gpu_utils import get_gpu_sm_version
from triton.testing import perf_report, do_bench, Benchmark, do_bench_cudagraph

try:
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
    HAVE_TRANSFORMER_ENGINE = True
except ImportError:
    HAVE_TRANSFORMER_ENGINE = False
    
try:
    from torch import _grouped_mm
    HAVE_TORCH_GROUPED_MM = True
except ImportError:
    HAVE_TORCH_GROUPED_MM = False
    
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
    line_names = ["torch.bmm", "Grouped-only", "Grouped+Gather", "Grouped+Scatter"]
    if HAVE_TRANSFORMER_ENGINE:
        line_vals.append("transformer-engine")
        line_names.append("TE Grouped Linear")
    # as of 2.9.0, torch._grouped_mm only supports hopper or newer.
    if HAVE_TORCH_GROUPED_MM and get_gpu_sm_version() >= 90:
        line_vals.append("torch-grouped-mm")
        line_names.append("Torch Grouped MM")
    
    return Benchmark(
        x_names=["num_tokens"],
        x_vals=list(range(256, 17000 + 1, 512)),
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

configs = []
configs.append(
    m_grouped_gemm_benchmark_extensive(
        "Qwen3-30B-A3B-style GEMM2, perfect routing",
        routing_func=perfect_routing,
        num_groups=128,
        N=2048,
        K=768,
        topk=8,
        dtype=torch.bfloat16
    ))
configs.append(
    m_grouped_gemm_benchmark_extensive(
        "OLMoE-1B-7B-style GEMM2, perfect routing",
        routing_func=perfect_routing,
        num_groups=64,
        N=2048,
        K=1024,
        topk=8,
        dtype=torch.bfloat16
    ))
configs.append(
    m_grouped_gemm_benchmark_extensive(
        "OLMoE-1B-7B-style GEMM2, balanced routing",
        routing_func=random_routing,
        num_groups=64,
        N=2048,
        K=1024,
        topk=8,
        dtype=torch.bfloat16
    ))
configs.append(
    m_grouped_gemm_benchmark_extensive(
        "Qwen3-30B-A3B-style GEMM2, balanced routing",
        routing_func=random_routing,
        num_groups=128,
        N=2048,
        K=768,
        topk=8,
        dtype=torch.bfloat16
    ))
configs.append(
    m_grouped_gemm_benchmark_extensive(
        "OLMoE-1B-7B-style GEMM2, skewed routing",
        routing_func=partial(random_skewed_routing, num_skewed_experts=8, skew_factor=16),
        num_groups=64,
        N=2048,
        K=1024,
        topk=8,
        dtype=torch.bfloat16
    ))
configs.append(
    m_grouped_gemm_benchmark_extensive(
        "Qwen3-30B-A3B-style GEMM2, skewed routing",
        routing_func=partial(random_skewed_routing, num_skewed_experts=8, skew_factor=16),
        num_groups=128,
        N=2048,
        K=768,
        topk=8,
        dtype=torch.bfloat16
    ))


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
    # do_bench_cuda_graph is also an option, but doesn't clear
    # the l2-cache. So, I think do_bench is a better option for this.
    return do_bench(lambda: f(), quantiles=quantiles)# warmup=150, rep=300)
    #return do_bench_cudagraph(lambda: f(), quantiles=quantiles, rep=100)

dist = torch.randn
#dist = torch.zeros

def benchmark_m_grouped_gemm_only(num_tokens, num_groups, N, K, topk, dtype, p):
    num_tokens = num_tokens * topk
    input = dist((num_tokens, K), device=torch.device("cuda"), dtype=dtype) 
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) * 0.023
    params = MGroupedGEMMParams(
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None
    )
    f = torch.compile(m_grouped_gemm, fullgraph=True)
    f(input, weight, p.group_indices, params)
    return bench(lambda: f(input, weight, p.group_indices, params))

def benchmark_m_grouped_gemm_gather(num_tokens, num_groups, N, K, topk, dtype, p):
    input = dist((num_tokens, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) * 0.023
    params = MGroupedGEMMParams(
        permute_indices=p.indices,
        gather=True,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None
    )
    f = torch.compile(m_grouped_gemm, fullgraph=True)
    f(input, weight, p.group_indices, params)
    return bench(lambda: f(input, weight, p.group_indices, params))

def benchmark_m_grouped_gemm_scatter(num_tokens, num_groups, N, K, topk, dtype, p, topk_scores):
    num_tokens_times_top = num_tokens * topk
    input = dist((num_tokens_times_top, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) * 0.023
    params = MGroupedGEMMParams(
        permute_indices=p.indices,
        gather=False,
        scatter=True,
        num_tokens=num_tokens,
        topk=topk,
        scales=topk_scores
    )
    f = torch.compile(m_grouped_gemm, fullgraph=True)
    f(input, weight, p.group_indices, params)
    return bench(lambda: f(input, weight, p.group_indices, params))

def benchmark_gemm_reference(num_tokens, num_groups, N, K, topk, dtype):
    r"""
    This is just a normal GEMM with the same number of FLOPs to other benchmarks
    benchmarks. This is just used a reference for performance.
    """
    num_tokens_times_topk = num_tokens * topk
    assert num_tokens % num_groups == 0
    input = dist((num_groups, num_tokens_times_topk // num_groups, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) * 0.023
    f = torch.bmm
    f(input, weight)
    return bench(lambda: f(input, weight))

def benchmark_te_grouped_linear(num_tokens, num_groups, N, K, topk, dtype, p):
    num_tokens = num_tokens * topk
    input = dist((num_tokens, K), device=torch.device("cuda"), dtype=dtype)
    m_splits = (p.group_indices[1:] - p.group_indices[:-1]).tolist()
    grouped_linear = GroupedLinear(num_groups, K, N, bias=False, params_dtype=dtype)
    return bench(lambda: grouped_linear(input, m_splits=m_splits, is_first_microbatch=None))

def benchmark_torch_grouped_mm(num_tokens, num_groups, N, K, topk, dtype, p):
    num_tokens_times_topk = num_tokens * topk
    input = dist((num_tokens_times_topk, K), device=torch.device("cuda"), dtype=dtype)
    weight = dist((num_groups, K, N), device=torch.device("cuda"), dtype=dtype) * 0.023
    offsets = p.group_indices[1:]
    print(input.size(), p.group_indices.size(), p.group_indices)
    func = torch.compile(_grouped_mm, fullgraph=True)
    func(input, weight, offs=offsets)
    try:
        return bench(lambda: func(input, weight, offs=offsets))
    except:
        print("torch grouped mm needs at least sm90")
        return 0, 0, 0

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
    torch._dynamo.reset()
    torch.manual_seed(0)

    if provider == "gemm-reference":
        ms, _, _ = benchmark_gemm_reference(num_tokens, num_groups, N, K, topk, dtype)
        
    # We need to ensure that each of these uses the same routing setup. 
    if provider in ("grouped-only", "grouped-gather", "grouped-scatter", "transformer-engine", "torch-grouped-mm"):
        topk_scores, topk_indices = routing_func(num_tokens, num_groups, topk, device="cuda", dtype=dtype)
        p = get_token_indices(
            topk_indices.view(-1),
            topk,
            num_groups,
            zero_prefix=True
        )
        if provider == "transformer-engine":
            ms, _, _ = benchmark_te_grouped_linear(num_tokens, num_groups, N, K, topk, dtype, p)
        if provider == "grouped-only":
            ms, _, _ = benchmark_m_grouped_gemm_only(num_tokens, num_groups, N, K, topk, dtype, p)
        if provider == "grouped-gather":
            ms, _, _ = benchmark_m_grouped_gemm_gather(num_tokens, num_groups, N, K, topk, dtype, p)
        if provider == "grouped-scatter":
            ms, _, _ = benchmark_m_grouped_gemm_scatter(num_tokens, num_groups, N, K, topk, dtype, p, topk_scores)
        if provider == "torch-grouped-mm":
            ms, _, _ = benchmark_torch_grouped_mm(num_tokens, num_groups, N, K, topk, dtype, p)
            
    flops = grouped_flops(num_tokens, num_groups, N, K, topk)
    tflops = flops / (ms / 1000) * 1e-12
    return tflops

benchmark_m_grouped_gemm_forward.run(print_data=True, save_path="./")