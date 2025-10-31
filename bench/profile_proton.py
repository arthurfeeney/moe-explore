import math
import torch
from moe_explore.functional.topk_moe import topk_moe
from moe_explore.params import MOEParams
from moe_explore.testing import random_topk_router, random_interleaved_glu
from moe_explore.router import router
from moe_explore.baseline.huggingface import make_huggingface_moe, set_huggingface_moe_weights
#from moe_explore.baseline.transformer_engine import TransformerEngineMoE
import triton.profiler as proton
import triton

seq_len = 32000
hidden_dim = 2048
intermediate_dim = 1024
num_experts = 64
topk = 8
activation = "swiglu"

def glu_interleaved_inputs():
    router_params = random_topk_router(num_experts, hidden_dim, topk, True, False, "cuda", torch.bfloat16)
    expert_params = random_interleaved_glu(
        num_experts,
        hidden_dim,
        intermediate_dim,
        activation,
        "cuda",
        torch.bfloat16,
    )
    moe_params = MOEParams(router_params, expert_params, num_experts=num_experts, topk=topk)
    input = torch.randn((seq_len, hidden_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return moe_params, input

session_id = proton.start(name="moe", context="shadow")

# warmup and autotuning
proton.deactivate()

# warmup gpu
moe_params, input = glu_interleaved_inputs()
for i in range(2):
    topk_moe(input, moe_params)

# Clear cache after warmup and in between iterations
cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
triton.runtime.driver.active.clear_cache(cache)
    
# Profile out fused moe
moe_params, input = glu_interleaved_inputs()
triton.runtime.driver.active.clear_cache(cache)
for i in range(3):
    proton.activate()
    with proton.scope("fused-interleaved"):
        output = topk_moe(input, moe_params)
        output.sum().backward()
    proton.deactivate()
    triton.runtime.driver.active.clear_cache(cache)
    
    
# setup huggingface moe
moe = make_huggingface_moe(hidden_dim, intermediate_dim, num_experts, topk, activation, torch.bfloat16)
moe = moe.to("cuda").to(torch.bfloat16)
set_huggingface_moe_weights(
    moe, 
    moe_params.router_params.router_weight, 
    moe_params.expert_params.weight1, 
    moe_params.expert_params.weight2
)

for i in range(3):
    proton.activate()
    with proton.scope("huggingface"):
        o1, o2 = moe(input.unsqueeze(0))
        o1.sum().backward()
    proton.deactivate()
    triton.runtime.driver.active.clear_cache(cache)
    
# setup transformer engine moe
"""
transformer_engine_moe = TransformerEngineMoE(
    hidden_dim, intermediate_dim, num_experts, topk, activation, torch.bfloat16)
transformer_engine_moe = transformer_engine_moe.to("cuda").to(torch.bfloat16)
transformer_engine_moe.init_weights(
    moe_params.expert_params.weight1, 
    moe_params.expert_params.weight2
)

for i in range(3):
    proton.activate()
    with proton.scope("transformer-engine"):
        router_func = lambda x: router(x, moe_params.router_params)
        moe(input.unsqueeze(0), router_func)
    proton.deactivate()
    triton.runtime.driver.active.clear_cache(cache)
"""
proton.finalize()