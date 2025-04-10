import torch
from scattermoe.mlp import MLP
from scattermoe.parallel_experts import ParallelExperts
from scattermoe import kernels

#BATCH = 16
#TOPK = 4
#SEQ_LEN = 1024
#EMBED_DIM = 512
#HIDDEN_DIM = 2048
#NUM_EXPERTS = 16
#device = "cuda"
#dtype = torch.float16

# DeepSeek-like sizes
BATCH = 16
TOPK = 6
SEQ_LEN = 4096
EMBED_DIM = 4096
HIDDEN_DIM = 4096
NUM_EXPERTS = 16
device = "cuda"
dtype = torch.float16

def run_parallel_experts_forward(
    BATCH,
    SEQ_LEN,
    EMBED_DIM,
    TOPK,
    NUM_EXPERTS,
    device,
    dtype
):    
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
    
    #for i in range(5):
    #    experts(X, k_weights, k_idxs)
    
    #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:    
    #    torch.cuda.profiler.cudart().cudaProfilerStart()
    #    experts(X, k_weights, k_idxs)
    #    torch.cuda.profiler.cudart().cudaProfilerStop()
    #prof.export_chrome_trace("scattermoe_forward.json")

    # do autotuning before profiling
    pe(x, TOPK, sorted_expert_idxs, sorted_scattered_idxs, 
       padded_block_idxs, expert_offsets, grouped_out=True)

    torch.cuda.profiler.cudart().cudaProfilerStart()
    pe(x, TOPK, sorted_expert_idxs, sorted_scattered_idxs, 
       padded_block_idxs, expert_offsets, grouped_out=True)   
    torch.cuda.profiler.cudart().cudaProfilerStop()
    
run_parallel_experts_forward(BATCH, SEQ_LEN, EMBED_DIM, TOPK, NUM_EXPERTS, device, dtype)