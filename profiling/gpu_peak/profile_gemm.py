r"""
This should be run with nsight compute. It is only intended to get
peak metrics for a roofline plot.
"""

import torch

def run_matmul(
    M,
    N,
    K,
    device,
    dtype
):    
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=dtype)
    
    torch.cuda.profiler.cudart().cudaProfilerStart()
    torch.matmul(a, b, out=c)
    torch.cuda.profiler.cudart().cudaProfilerStop()
    
# Note: for float16 matmul, pytorch accumulates with float32 
run_matmul(8 * 2048, 1024, 1024, 'cuda', torch.float16)