import torch
from transformers import AutoConfig, AutoModelForCausalLM

from moe_explore.moe import TopkMoE
from moe_explore.baseline.torch_grouped_mm import TorchGroupedMMMoE
from moe_explore.baseline.scattermoe import make_scattermoe_mlp, scattermoe_forward
from moe_explore.router import router
from moe_explore.params import TopkRouterParams

def load_model_with_topk_moe(model_name: str):
    config = AutoConfig.from_pretrained(model_name, cache_dir="/pub/afeeney/huggingface/cache/")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        cache_dir="/pub/afeeney/huggingface/cache/"
    ).to("cuda").to(torch.bfloat16)
    n_layers = len(model.model.layers)
    with torch.no_grad():
        for i in range(n_layers):
            moe = TopkMoE(
                config.num_experts,
                config.hidden_size,
                config.intermediate_size,
                config.num_experts_per_tok,
                # olmoe config defaults to silu, so manually set as swiglu
                "swiglu",
                return_topk_logits=True
            ).to("cuda").to(torch.bfloat16)
            moe = torch.compile(moe, fullgraph=True)
            
            mlp = model.model.layers[i].mlp
            moe.router_weight[:] = mlp.gate.weight.data.t()
            for j in range(config.num_experts):
                moe.weight1[j, :, 0::2] = mlp.experts[j].gate_proj.weight.data.t()
                moe.weight1[j, :, 1::2] = mlp.experts[j].up_proj.weight.data.t()
                moe.weight2[j] = mlp.experts[j].down_proj.weight.data.t()   
            
            # Remove the original mlp, and set the new one
            del model.model.layers[i].mlp
            model.model.layers[i].mlp = moe   
    return model

def load_model_with_torch_moe(model_name: str):
    config = AutoConfig.from_pretrained(model_name,  cache_dir="/pub/afeeney/huggingface/cache/")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        cache_dir="/pub/afeeney/huggingface/cache/"
    ).to("cuda").to(torch.bfloat16)
    n_layers = len(model.model.layers)
    with torch.no_grad():
        for i in range(n_layers):
            mlp = model.model.layers[i].mlp
            
            # Make a dummy class so that we can use our own router
            class DumbMoE(TorchGroupedMMMoE):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.router_params = TopkRouterParams(
                        router_weight=mlp.gate.weight.data.t(),
                        topk=config.num_experts_per_tok,
                        softmax_before_topk=True,
                        normalize_routing=False
                    )
                    
                def forward(self, input: torch.Tensor):
                    shape = input.shape
                    # flatten batch and seq dims
                    input = input.view(-1, input.size(-1))
                    output, router_logits = super().forward(input, lambda x: router(x, self.router_params))
                    return output.view(shape), router_logits
            
            moe = DumbMoE(
                config.hidden_size,
                config.intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
                # olmoe config defaults to silu, so manually set as swiglu
                "swiglu",
                torch.bfloat16
                #return_topk_logits=True
            ).to("cuda").to(torch.bfloat16)
            moe = torch.compile(moe, fullgraph=True)
                        
            for j in range(config.num_experts):
                moe.weight1[j] = mlp.experts[j].gate_proj.weight.data.t()
                moe.weight2[j] = mlp.experts[j].up_proj.weight.data.t()
                moe.weight3[j] = mlp.experts[j].down_proj.weight.data.t()
                
            # Remove the original mlp, and set the new one
            del model.model.layers[i].mlp
            model.model.layers[i].mlp = moe
    return model

def load_model_with_scattermoe(model_name: str):
    config = AutoConfig.from_pretrained(model_name,  cache_dir="/pub/afeeney/huggingface/cache/")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        cache_dir="/pub/afeeney/huggingface/cache/"
    ).to("cuda").to(torch.bfloat16)
    n_layers = len(model.model.layers)
    with torch.no_grad():
        for i in range(n_layers):
            mlp = model.model.layers[i].mlp
            
            # Make a dummy class so that we can use our own router
            class DumbMoE(TorchGroupedMMMoE):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.router_params = TopkRouterParams(
                        router_weight=mlp.gate.weight.data.t(),
                        topk=config.num_experts_per_tok,
                        softmax_before_topk=True,
                        normalize_routing=False
                    )
                    
                def forward(self, input: torch.Tensor):
                    shape = input.shape
                    # flatten batch and seq dims
                    input = input.view(-1, input.size(-1))
                    output, router_logits = super().forward(input, lambda x: router(x, self.router_params))
                    return output.view(shape), router_logits
            
            moe = DumbMoE(
                config.hidden_size,
                config.intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
                # olmoe config defaults to silu, so manually set as swiglu
                "swiglu",
                torch.bfloat16
                #return_topk_logits=True
            ).to("cuda").to(torch.bfloat16)
            moe = torch.compile(moe)
                        
            for j in range(config.num_experts):
                moe.weight1[j] = mlp.experts[j].gate_proj.weight.data.t()
                moe.weight2[j] = mlp.experts[j].up_proj.weight.data.t()
                moe.weight3[j] = mlp.experts[j].down_proj.weight.data.t()
                
            # Remove the original mlp, and set the new one
            del model.model.layers[i].mlp
            model.model.layers[i].mlp = moe
    return model


#model = load_model_with_topk_moe("allenai/OLMoE-1B-7B-0924").to("cuda").to(torch.bfloat16)
#model = load_model_with_torch_moe("allenai/OLMoE-1B-7B-0924").to("cuda").to(torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924",
    dtype=torch.bfloat16,
    cache_dir="/pub/afeeney/huggingface/cache/"
).to("cuda").to(torch.bfloat16)

# The model as a whole cannot use fullgraph=True, but
# the moe layers are (above). However, fullgraph=True or not
# seems to not have much effect on memory usage here.

# On A100,
# topk moe can use batch size of 33
# torch moe can use batch size of 28
#input = torch.randint(0, 1000, (28, 1024), device="cuda", dtype=torch.int32)
# default hf can use batch size of 22
#input = torch.randint(0, 1000, (22, 1024), device="cuda", dtype=torch.int32)

torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 16

# dynamic=False, so the model is fully recompiled for each batch size
# huggingface needs dynamic=True, since the experts each handle different sized inputs
#model = torch.compile(model, dynamic=False)
#model = torch.compile(model)

max_batch_size = 22
#for batch_size in range(20, 21, 2):

for batch_size in range(2, max_batch_size + 1, 2):
    #torch._dynamo.reset()    
    
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    
    input = torch.randint(0, 1000, (batch_size, 1024), device="cuda", dtype=torch.int32)
    
    # compile forward and backward pass before timing
    output = model(input)
    loss = output.logits.sum()
    loss.backward()
    del loss, output
    model.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    total_time = 0
    iters = 10
    for i in range(iters):
        # Time forward and backward pass
        start.record()
        output = model(input)
        loss = output.logits.sum()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        dur = start.elapsed_time(end)
        total_time += dur

        # Reset things so there's enough memory for next iteration
        del loss, output
        model.zero_grad(set_to_none=True)
        
    avg_seconds = (total_time / iters) / 1000
    print(f"batch_size: {batch_size}, tokens / second: ", (input.shape[0] * input.shape[1]) / avg_seconds)


"""
print("forward")
free, total = torch.cuda.memory.mem_get_info()
print((total - free) / (1024 ** 3), free, total)
output = model(input)

torch.cuda.synchronize()
print("backward")
free, total = torch.cuda.memory.mem_get_info()
print((total - free) / (1024 ** 3), free, total)

# Note: it's impossible to do backward on some GPUs. I.e., A30 just doesn't have enough memory.
loss = output.logits.sum()
loss.backward()

torch.cuda.synchronize()
print("done")
free, total = torch.cuda.memory.mem_get_info()
print((total - free) / (1024 ** 3), free, total)

print("Loss should be around the same for differnet Implementations:")
print("loss", loss)
"""