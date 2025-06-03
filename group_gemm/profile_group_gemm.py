import torch
from group_gemm_triton import group_gemm_fn
import triton

num_groups = 8
group_A = [torch.randn((2048, 2048), device="cuda", dtype=torch.float16) for _ in range(num_groups)]
group_B = [torch.randn((2048, 2048), device="cuda", dtype=torch.float16) for _ in range(num_groups)]

# do autotuning.
c = group_gemm_fn(group_A, group_B)

torch.cuda.profiler.cudart().cudaProfilerStart()
c = group_gemm_fn(group_A, group_B)
torch.cuda.profiler.cudart().cudaProfilerStop()