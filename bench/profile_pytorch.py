import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity
from moe_explore.functional.topk_moe import topk_moe
from moe_explore.router import router
from moe_explore.params import MOEParams
from moe_explore.testing import random_topk_router, random_mlp, random_interleaved_glu
from moe_explore.baseline.torch_grouped_mm import TorchGroupedMMMoE

# olmoe sizes
num_tokens = 32768
hidden_dim = 2048
intermediate_dim = 1024
num_experts = 64
topk = 8
activation = "swiglu"

input = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)

router_params = random_topk_router(num_experts, hidden_dim, topk, True, False, "cuda", torch.bfloat16)
expert_params = random_interleaved_glu(
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

topk_moe_compiled = torch.compile(topk_moe, fullgraph=True)

moe = TorchGroupedMMMoE(hidden_dim, intermediate_dim, num_experts, topk, "swiglu", torch.bfloat16).to("cuda").to(torch.bfloat16)
moe.init_weights(moe_params.expert_params.weight1, moe_params.expert_params.weight2)
moe = torch.compile(moe, fullgraph=True)
router = torch.compile(router, fullgraph=True)

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
    #with torch.inference_mode():
    for i in range(7):
        torch.compiler.cudagraph_mark_step_begin()
        with record_function("topk_moe_forward"):
            #output = topk_moe_compiled(input, moe_params)
            output, _ = moe(input, lambda x: router(x, router_params))
        with record_function("topk_moe_backward"):
            output.sum().backward()
        torch.cuda.synchronize()
        prof.step()
    
prof.export_chrome_trace("trace.json")