import argparse
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from transformers import (
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer
)
import torch
import torch.nn.functional as F

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
        required=True
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPUs: {torch.cuda.device_count()}")

    model_name = _MODEL_NAME_DICT[args.model_name]
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=args.hf_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=args.hf_cache_dir,
            low_cpu_mem_usage=True,
            output_router_logits=True,
            device_map="auto"
    )
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = tokenized_wikitext(tokenizer)

    seq_len = min(1000, tokenized_dataset.input_ids.size(1))
    stride = 64

    model.eval()
    with torch.no_grad():
        expert_counts = torch.zeros(model_config.num_hidden_layers, model_config.num_experts, device=device)
        for step in range(0, seq_len, stride):
            print(f"step {step}")
            input_ids = tokenized_dataset.input_ids[..., step:step + stride]
            output = model(input_ids=input_ids.to(device), output_router_logits=True)
            for layer_idx in range(model_config.num_hidden_layers):
                selected_experts = router(output.router_logits[layer_idx], topk=model_config.num_experts_per_tok)
                count = torch.bincount(selected_experts.view(-1), minlength=model_config.num_experts)
                expert_counts[layer_idx] += count 

    to_save = {
        "expert_counts": expert_counts.detach().cpu(),
        "num_tokens": seq_len,
        "stride": stride,
        "model_name": args.model_name,
        "dataset_name": "wikitext"
    }
    torch.save(to_save, f"wikitext_counts_{args.model_name}.pt")

def tokenized_wikitext(tokenizer):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    assert isinstance(dataset, DatasetDict)
    assert "test" in dataset
    assert "train" in dataset
    assert "validation" in dataset

    token_dataset = dataset["test"]
    # This loads the entire wikitext dataset.
    # The .join is based from https://huggingface.co/docs/transformers/perplexity
    tokens = tokenizer("\n\n".join(token_dataset["text"]), return_tensors="pt")
    assert tokens.input_ids.dim() == 2
    return tokens

def router(router_logits, topk):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    _, selected_experts = torch.topk(routing_weights, topk, dim=-1)
    return selected_experts
        
if __name__ == "__main__":
    main()
