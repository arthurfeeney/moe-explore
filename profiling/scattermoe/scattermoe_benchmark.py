r"""
This script is just intended to get a sense of where the performance is relative to GPU peak.
A30 theoretical peak with float16 is 165 teraflops.
"""

import torch
from triton.testing import Benchmark, do_bench, perf_report
from scattermoe.mlp import MLP
from scattermoe.parallel_experts import ParallelExperts
from scattermoe import kernels

BATCH = 16
TOPK = 4
EMBED_DIM = 2048
HIDDEN_DIM = EMBED_DIM
NUM_EXPERTS = 8

configs = []
configs.append(
    Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[128, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["parallel-linear", "linear", "a100-peak"],
        line_names=["Parallel Linear", "Linear", "A100 [peak]"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="scattermoe-forward-tflops",
        args={
            "BATCH": BATCH,
            "TOPK": TOPK,
            "EMBED_DIM": EMBED_DIM,
            "NUM_EXPERTS": NUM_EXPERTS,
            "device": "cuda",
            "dtype": torch.float16
        }
    )
)

@perf_report(configs)
def benchmark_parallel_experts_forward(
    BATCH,
    SEQ_LEN,
    EMBED_DIM,
    TOPK,
    NUM_EXPERTS,
    provider,
    device,
    dtype
):  
    if provider == "a100-peak":
        return 165
    if provider == "linear":
        A = torch.randn(BATCH * SEQ_LEN * TOPK, EMBED_DIM, device=device, dtype=dtype)
        B = torch.randn(EMBED_DIM, HIDDEN_DIM, device=device, dtype=dtype)      
        ms = do_bench(lambda: A @ B)
        flops = 2 * A.size(0) * A.size(1) * B.size(1)
        tflops = (flops * 1e-12) / (ms * 1e-3)
        return tflops
    
    logits = torch.randn(BATCH * SEQ_LEN, EMBED_DIM, device=device, dtype=dtype)
    weights = torch.softmax(logits.float(), axis=-1).to(device).to(dtype)
    x = torch.randn(BATCH * SEQ_LEN, EMBED_DIM, dtype=dtype, device=device, requires_grad=True)
    k_weights, k_idxs = torch.topk(weights, TOPK)
    k_weights.requires_grad_()
    
    x_shape = x.size()
    x = x.view(-1, x_shape[-1])
    with torch.no_grad():
        sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(k_idxs)
        padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, NUM_EXPERTS)

    pe = ParallelExperts(
        num_experts=NUM_EXPERTS,
        input_size=EMBED_DIM,
        output_size=HIDDEN_DIM
    ).to(dtype).to(device)
    
    ms = do_bench(lambda: pe(x, TOPK, sorted_expert_idxs, sorted_scattered_idxs, 
                             padded_block_idxs, expert_offsets, grouped_out=True))
    
    flops = 2 * (BATCH * SEQ_LEN * TOPK * EMBED_DIM * HIDDEN_DIM)
    tflops = flops * 1e-12 / (ms * 1e-3)
    return tflops

benchmark_parallel_experts_forward.run(save_path=".", print_data=True)