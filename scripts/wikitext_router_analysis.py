import argparse
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer
)
import torch
import torch.nn.functional as F

OLMOE = "olmoe"
QWEN3 = "qwen3"
ERNIE4 = "ernie4"
OLMOE_HF = "allenai/OLMoE-1B-7B-0924"
QWEN3_HF = "Qwen/Qwen3-30B-A3B"
ERNIE4_HF = "baidu/ERNIE-4.5-21B-A3B-Base-PT"

_MODEL_NAME_DICT = {
    OLMOE: OLMOE_HF,
    QWEN3: QWEN3_HF,
    ERNIE4: ERNIE4_HF
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", 
        "--model-name", 
        type=str, 
        required=True,
        choices=[OLMOE, QWEN3, ERNIE4]
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        required=True
    )
    args = parser.parse_args()
    
    print(args.model_name)
    print(args.hf_cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPUs: {torch.cuda.device_count()}")

    model_name = _MODEL_NAME_DICT[args.model_name]
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=args.hf_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=args.hf_cache_dir,
            low_cpu_mem_usage=False,
            output_router_logits=True,
            device_map="auto"
    )
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.hf_cache_dir)
    tokenized_dataset = tokenized_wikitext(tokenizer, args.hf_cache_dir)

    print(tokenized_dataset.input_ids.size())

    num_tokens = min(500000, tokenized_dataset.input_ids.size(1))
    print(f"num_tokens: {num_tokens}")
    seq_len = 512

    model.eval()
    with torch.no_grad():
        expert_counts = torch.zeros(model_config.num_hidden_layers, model_config.num_experts, device="cpu")
        for step in range(0, num_tokens, seq_len):
            print(f"step {step}")
            input_ids = tokenized_dataset.input_ids[..., step:step + seq_len]
            output = model(input_ids=input_ids, output_router_logits=True)
            for layer_idx in range(model_config.num_hidden_layers):
                selected_experts = router(output.router_logits[layer_idx], topk=model_config.num_experts_per_tok).to("cpu")
                count = torch.bincount(selected_experts.view(-1), minlength=model_config.num_experts)
                expert_counts[layer_idx] += count 

    to_save = {
        "expert_counts": expert_counts.detach().cpu(),
        "num_tokens": num_tokens,
        "seq_len": seq_len,
        "model_name": args.model_name,
        "dataset_name": "wikitext"
    }
    torch.save(to_save, f"wikitext_counts_{args.model_name}.pt")

def tokenized_wikitext(tokenizer, cache_dir):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=cache_dir)
    assert isinstance(dataset, DatasetDict)
    assert "test" in dataset
    assert "train" in dataset
    assert "validation" in dataset

    token_dataset = dataset["train"]
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
