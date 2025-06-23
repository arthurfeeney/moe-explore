r"""
This should be run with nsight compute to get a roofline model.
In ParallelExperts, we only care about the scatter2scatter kernel.
"""

import torch
from scattermoe import kernels
from scattermoe.parallel_experts import ParallelExperts

def run_matmul(
    M,
    N,
    K,
    device,
    dtype
):
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    fn = torch.compile(torch.matmul)    
    c = fn(a, b)
    
    torch.cuda.profiler.cudart().cudaProfilerStart()
    c = fn(a, b)
    torch.cuda.profiler.cudart().cudaProfilerStop()
    
def run_scatter(M, N, K, num_experts, topk_idxs, topk, device, dtype):
    x = torch.randn((M, K),
                    dtype=dtype, 
                    device=device,
                    requires_grad=True)
    
    pe = ParallelExperts(
        num_experts=num_experts,
        input_size=K,
        output_size=N
    ).to(dtype).to(device)

    # The sorting technically should be timed as well, but excluding it because ncu
    # can't really account for time of multiple kernels.
    x_shape = x.size()
    x = x.view(-1, x_shape[-1])
    with torch.no_grad():
        sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(topk_idxs)
        padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, num_experts)
    
    # call once for jit-compiling and autotuning
    # ncu should flush cache
    pe(x, 
        topk, 
        sorted_expert_idxs, 
        sorted_scattered_idxs, 
        padded_block_idxs, 
        expert_offsets, 
        grouped_out=True)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    torch.cuda.profiler.cudart().cudaProfilerStart()
    pe(x, 
        topk, 
        sorted_expert_idxs, 
        sorted_scattered_idxs, 
        padded_block_idxs, 
        expert_offsets, 
        grouped_out=True)
    torch.cuda.profiler.cudart().cudaProfilerStop()
    
def topk_idxs_all_zero(M, num_experts, topk, device):
    r"""
    route everything to expert 0, extremely imbalanced.
    """
    topk_idxs = torch.zeros((M, topk), device=device, dtype=torch.long)
    return topk_idxs

def topk_idxs_uniform(M, num_experts, topk, device):
    r"""
    Route such that each expert gets an even-ish amount of data.
    Ensures every expert is used, so arithmetic intensity should be lower.
    """
    topk_idxs = torch.arange(num_experts, device=device, dtype=torch.long).repeat(M * num_experts)[:M * topk].reshape(M, topk)
    return topk_idxs

device = 'cuda'
dtype = torch.float16

topk = 2

sizes = []
for m in [128, 512, 1024, 2048, 4096, 8192, 2 * 8192, 4 * 8192]:
    #for k in [128, 512, 1024, 2048, 4098]:
    k = 1024
    n = k
    sizes.append((m * topk, n, k))

for (m, n, k) in sizes:
    run_matmul(m, n, k, device, dtype)

# on smaller sizes, the flops/byte looks artficially large.
# I think this is because of the padding that triton has to do...
# it loads a very small bit of data, and pads it to the block size.
# then, the flops will be large relative to the amount of data loaded.
num_experts = 8
for (m, n, k) in sizes:
    topk_idxs = topk_idxs_uniform(m, num_experts, topk, device)
    # m-dimension is dvided because we effectively
    # multiply it by topk when routing to experts.
    run_scatter(m, n, k, num_experts, topk_idxs, topk, device, dtype)