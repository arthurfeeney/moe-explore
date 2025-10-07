import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity
from moe_explore.functional.topk_moe import topk_moe_forward
from moe_explore.params import MOEParams
from moe_explore.testing import random_topk_router, random_mlp

num_tokens = 32000
hidden_dim = 2048
intermediate_dim = 768
num_experts = 128
topk = 8
activation = "silu"

input = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)

router_params = random_topk_router(num_experts, hidden_dim, topk, True, False, "cuda", torch.bfloat16)
expert_params = random_mlp(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    "cuda",
    torch.bfloat16,
)

input.requires_grad = True
router_params.router_weight.requires_grad = True
expert_params.weight1.requires_grad = True
expert_params.weight2.requires_grad = True

moe_params = MOEParams(router_params, expert_params, num_experts=num_experts, topk=topk)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    with_stack=True,
    schedule=schedule(
        wait=1,
        warmup=1,
        active=5,
        repeat=1
    )
) as prof:
    
    for i in range(6):
        with record_function("topk_moe_forward"):
            output = topk_moe_forward(
                input,
                moe_params
            )
            
        with record_function("topk_moe_backward"):
            output.sum().backward()
            
        torch.cuda.synchronize()
        
        prof.step()
        
prof.export_chrome_trace("trace.json")