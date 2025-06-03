import torch
from group_gemm_triton import group_gemm_fn
import triton
from triton.testing import Benchmark, do_bench, perf_report

configs = []
configs.append(
    Benchmark(
        x_names=["M"],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["group", "torch", "peak"],
        line_names=["Triton Group GEMM", "Torch GEMM", "A30 [peak]"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="scattermoe-forward-tflops",
        args={
            "num_groups": 8,
            "N": 4096,
            "K": 4096,
            "device": "cuda",
            "dtype": torch.float16
        }
    )
)

@perf_report(configs)
def benchmark_parallel_experts_forward(
    num_groups,
    M, 
    N,
    K,
    provider,
    device,
    dtype
):
    quantiles = [0.2, 0.5, 0.8]
    if provider == "peak":
        device_name = torch.cuda.get_device_name()
        print(device_name)
        if "A100" in device_name:
            return 312
        if "A30" in device_name: 
           return 165
    elif provider == "group":
        group_A = [torch.randn((M, K), dtype=dtype, device=device) for _ in range(num_groups)]
        group_B = [torch.randn((K, N), dtype=dtype, device=device) for _ in range(num_groups)]
        ms, min_ms, max_ms = do_bench(lambda: group_gemm_fn(group_A, group_B), quantiles=quantiles)
        flops = num_groups * 2 * M * N * K
    elif provider == "torch":
        A = torch.randn((num_groups, M, K), dtype=dtype, device=device)
        B = torch.randn((num_groups, K, N), dtype=dtype, device=device)
        ms, min_ms, max_ms = do_bench(lambda: A @ B, quantiles=quantiles)
        flops = num_groups * 2 * M * N * K

    tflops = flops * 1e-12 / (ms * 1e-3)
    return tflops

benchmark_parallel_experts_forward.run(save_path=".", print_data=True)