import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoConfig, 
)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn

OLMOE = "olmoe"
QWEN3 = "qwen3"
OLMOE_HF = "allenai/OLMoE-1B-7B-0924"
QWEN3_HF = "Qwen/Qwen3-30B-A3B"

_MODEL_NAME_DICT = {
    OLMOE: OLMOE_HF,
    QWEN3: QWEN3_HF
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", 
        "--model-name", 
        type=str, 
        required=True,
        choices=[OLMOE, QWEN3]
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        required=False
    )
    args = parser.parse_args()

    model_name = _MODEL_NAME_DICT[args.model_name]
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="/pub/afeeney/huggingface/cache/",
            low_cpu_mem_usage=True
    )
    print(model)

    num_experts = model_config.num_experts
    topk = model_config.num_experts_per_tok

    layers = model.model.layers
    routing_weights = [layer.mlp.gate.weight for layer in layers]
    
    batch_size = 64
    seq_len = 1024
    tokens = torch.randn((batch_size, seq_len, model_config.hidden_size))
    
    with torch.no_grad():
        percentages = model_routing_percentages(routing_weights, tokens, topk, num_experts).cpu().numpy()

    for i in range(percentages.shape[0]):
        min_perc = percentages[i].min()
        max_perc = percentages[i].max()
        print(f"Layer {i}: min={min_perc}, max={max_perc}")

    ax = seaborn.heatmap(percentages)
    ax.set(xlabel="Expert", ylabel="Layer")
    plt.savefig(f"percentages_{args.model_name}.png")

def router(tokens, router_weight, topk):
    r"""
    huggingface models do not have a separate module for the router.
    Instead, there's just a SMoE class that does both routing and evaluates experts.
    This is essentially copy-pasted from huggingface.
    """
    router_logits = F.linear(tokens, router_weight)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, topk, dim=-1)
    return routing_weights, selected_experts

def layer_routing_percentages(tokens, router_weight, topk, num_experts):
    _, selected_experts = router(tokens, router_weight, topk)
    selected_experts = selected_experts.view(-1)
    expert_counts = torch.bincount(selected_experts, minlength=num_experts)
    expert_percentages = expert_counts / (tokens.size(0) * tokens.size(1))
    return expert_percentages

def model_routing_percentages(router_weights, tokens, topk, num_experts):
    percentages = []
    for router_weight in router_weights:
        expert_percentages = layer_routing_percentages(tokens, router_weight, topk, num_experts)
        percentages.append(expert_percentages)
    return torch.stack(percentages, dim=0)

if __name__ == "__main__":
    main()
