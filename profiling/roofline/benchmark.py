from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import triton
from triton.testing import do_bench

from scattermoe import kernels
from scattermoe.parallel_experts import ParallelExperts

@dataclass
class MoEConfig:
    batch_size: int
    seq_len: int
    embed_dim: int
    hidden_dim: int
    num_experts: int
    topk: int
    device: torch.device
    dtype: torch.dtype
    
@dataclass
class BenchmarkResults:
    fn_name: str
    tflops: float
    arith_intensity: float
    
def gemm_tflops(M, N, K, ms):
    return (2 * M * N * K * 1e-12) / (ms * 1e-3)

def gemm_ai(M, N, K):
    # This is an upper bound on achievable intensity. 
    return (M * N * K) / (M * K + N * K + M * N)

def moe_ai(M_sizes, N, K):
    # This is an upper bound on achievable intensity.
    # The denominator is different since we load stuff for each expert
    # depending on the number of tokens it gets.
    accum = 0
    for M in M_sizes:
        accum += (M * K + N * K + M * N)
    return ((M * N * K) / accum).cpu().item()
    
def benchmark_scatter_moe_forward(config):
    x = torch.randn((config.batch_size * config.seq_len, config.embed_dim),
                    dtype=config.dtype, 
                    device=config.device,
                    requires_grad=True)

    # setup an artifical routing
    logits = torch.randn(config.batch_size * config.seq_len, config.num_experts, device=config.device, dtype=config.dtype)
    routing_weights = torch.softmax(logits.float(), axis=-1).to(config.device).to(config.dtype)
    _, topk_idxs = torch.topk(routing_weights, config.topk)
    
    assert topk_idxs.min() >= 0
    assert topk_idxs.max() < config.num_experts
    
    pe = ParallelExperts(
        num_experts=config.num_experts,
        input_size=config.embed_dim,
        output_size=config.hidden_dim
    ).to(config.dtype).to(config.device)
    
    def kernel(x):
        # sorting should be timed because
        # it is actually a pretty large slow down for the kernel. Without the sorting,
        # it's a good bit faster, but still slower than a standard torch.matmul
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(topk_idxs)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, config.num_experts)
        pe(x, 
            config.topk, 
            sorted_expert_idxs, 
            sorted_scattered_idxs, 
            padded_block_idxs, 
            expert_offsets, 
            grouped_out=True)
        
    ms = do_bench(lambda: kernel(x))

    M_sizes = torch.bincount(topk_idxs.view(-1)).cpu()          
    ai = moe_ai(M_sizes=M_sizes,
                N=config.hidden_dim,
                K=config.embed_dim)
    
    return ms, ai
    
    
def benchmark_gemm_forward(config: MoEConfig):
    # This constructs a GEMM with similar problem size to the MOE.
    # This should set an upper bound on performance, and the arithmetic intensity
    # should predictably be about `num_experts` times larger.
    # M is scaled by topk because the MOE essentially expands by topk after scattering.
    M = config.batch_size * config.seq_len * config.topk
    N = config.hidden_dim
    K = config.embed_dim
    
    A = torch.randn((M, K), device=config.device, dtype=config.dtype)
    B = torch.randn((K, N), device=config.device, dtype=config.dtype)
    
    fn = torch.matmul
    
    ms = do_bench(lambda: fn(A, B))
    print(torch.cuda.clock_rate())

    ai = gemm_ai(M=config.batch_size * config.seq_len * config.topk,
                 N=config.hidden_dim,
                 K=config.embed_dim)
    
    return ms, ai
    
def benchmark_grouped_gemm_forward():
    pass

def benchmark_MNK(fn, dtype):
    tflops = {}
    arith_intensity = {}
    #for MNK in [128, 256, 512, 1024, 2048, 4096, 8192]:
    for MNK in range(1024, 8192):
        config = MoEConfig(
            batch_size=1,
            seq_len=MNK,
            embed_dim=MNK,
            hidden_dim=MNK,
            num_experts=8,
            topk=2,
            device='cuda',
            dtype=dtype
        )
        ms, ai = fn(config)
        tflops[MNK] = gemm_tflops(M=config.batch_size * config.seq_len * config.topk,
                                  N=config.hidden_dim,
                                  K=config.embed_dim,
                                  ms=ms)
        arith_intensity[MNK] = ai
    return tflops, arith_intensity
        
scatter_tflops, scatter_ai = benchmark_MNK(benchmark_scatter_moe_forward, torch.float16)
matmul_tflops, matmul_ai = benchmark_MNK(benchmark_gemm_forward, torch.float16)

print(scatter_tflops)
print(scatter_ai)
print()
print(matmul_tflops)
print(matmul_ai)

a30_float16_peak_tflops_per_sec = 165
a30_float16_peak_tbytes_per_sec = 0.933086
machine_balance = a30_float16_peak_tflops_per_sec / a30_float16_peak_tbytes_per_sec 


num_samples = len(scatter_tflops)
plt.plot(scatter_tflops.keys(), [a30_float16_peak_tflops_per_sec]*num_samples, label='A30 Peak', linestyle='--')
plt.scatter(scatter_tflops.keys(), scatter_tflops.values(), label='scatter', marker='x', c='b')
plt.scatter(matmul_tflops.keys(), matmul_tflops.values(), label='torch.matmul', marker='x', c='r')
plt.xscale('log')
plt.xlabel('Matrix dimensions (MNK)')
plt.ylabel('TFLOPS/s')
plt.legend()
plt.savefig("bench.png")
plt.close()

#plt.ylabel('tflops')
#plt.xlabel('matrix size')
#plt.legend()
#plt.savefig('bench.png')
#plt.close()
