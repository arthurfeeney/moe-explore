import lm_eval
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from moe_explore.moe import TopkMoE

def load_model_with_topk_moe(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16
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
            
            mlp = model.model.layers[i].mlp
            moe.router_weight[:] = mlp.gate.weight.data.t()
            for j in range(config.num_experts):
                moe.weight1[j, :, 0::2] = mlp.experts[j].gate_proj.weight.data.t()
                moe.weight1[j, :, 1::2] = mlp.experts[j].up_proj.weight.data.t()
                moe.weight2[j] = mlp.experts[j].down_proj.weight.data.t()   
            model.model.layers[i].mlp = moe
    return model

#model = load_model_with_topk_moe("allenai/OLMoE-1B-7B-0924")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", dtype=torch.bfloat16).to("cuda")

lm_model = lm_eval.models.huggingface.HFLM(model, backend="causal", batch_size=16)

task_manager = lm_eval.tasks.TaskManager()

results = lm_eval.simple_evaluate(
    model=lm_model,
    tasks=[
        "sciq",
        "hellaswag"
    ],
    num_fewshot=0,
    task_manager=task_manager
)

print(results.keys())
print(results["results"].keys())


if "sciq" in results["results"].keys():
    print("sciq:")
    print(results["results"]["sciq"])
    print("acc,none: ", results["results"]["sciq"]["acc,none"])
    print("acc_stderr,none: ", results["results"]["sciq"]["acc_stderr,none"])
    print("acc_norm,none: ", results["results"]["sciq"]["acc_norm,none"])
    print("acc_norm_stderr,none: ", results["results"]["sciq"]["acc_norm_stderr,none"])

if "hellaswag" in results["results"].keys():
    print("hellaswag:")
    print(results["results"]["hellaswag"])
    print("acc,none: ", results["results"]["hellaswag"]["acc,none"])
    print("acc_stderr,none: ", results["results"]["hellaswag"]["acc_stderr,none"])
    print("acc_norm,none: ", results["results"]["hellaswag"]["acc_norm,none"])
    print("acc_norm_stderr,none: ", results["results"]["hellaswag"]["acc_norm_stderr,none"])