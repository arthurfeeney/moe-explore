import torch
import pytest
import numpy as np
from moe_explore.hf_moe_reference import olmoe_forward, qwen3_moe_forward, ernie4_5_moe_forward
from moe_explore.functional.topk_moe import topk_moe_forward
from transformers import AutoConfig
from moe_explore.params import MOEParams, MLPParams
from moe_explore.testing import random_glu, random_topk_router, random_ernie_router, assert_close

OLMOE = "allenai/OLMoE-1B-7B-0924"
QWEN3 = "Qwen/Qwen3-30B-A3B"
ERNIE4 = "baidu/ERNIE-4.5-21B-A3B-Base-PT"

def hf_config_to_moe_params(config, model_name):
    expert_params = random_glu(
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
        config.hidden_act,
        "cuda",
        torch.bfloat16
    )    
    if model_name in (OLMOE, QWEN3):
        return MOEParams(
            router_params=random_topk_router(
                config.num_experts,
                config.hidden_size,
                config.num_experts_per_tok,
                softmax_before_topk=True,
                normalize_routing=config.norm_topk_prob,
                device="cuda",
                dtype=torch.bfloat16
            ),
            expert_params=expert_params,
            num_experts=config.num_experts,
            topk=config.num_experts_per_tok,
        )
    if model_name == ERNIE4:
        return MOEParams(
            router_params=random_ernie_router(
                config.moe_num_experts,
                config.hidden_size,
                config.moe_k,
                "cuda",
                torch.bfloat16
            ),
            expert_params=expert_params,
            num_experts=config.moe_num_experts,
            topk=config.moe_k,
        )
        
def get_interleave_glu_params(input, moe_params):
    size = (
        moe_params.num_experts, 
        moe_params.expert_params.gate_weight.size(1),
        2 * moe_params.expert_params.gate_weight.size(2)
    )
    interleaved_weight = torch.empty(size, device=input.device, dtype=input.dtype)
    interleaved_weight[:, :, 0::2] = moe_params.expert_params.gate_weight
    interleaved_weight[:, :, 1::2] = moe_params.expert_params.up_weight
    
    if moe_params.expert_params.activation == "gelu":
        activation = "geglu"
    else:
        activation = "swiglu"
    
    interleaved_glu_params = MLPParams(
        weight1=interleaved_weight,
        weight2=moe_params.expert_params.down_weight,
        activation=activation
    )
    moe_params.expert_params = interleaved_glu_params
    return moe_params

@pytest.mark.parametrize(
    "seq_len,model_name,forward", [(128, OLMOE, olmoe_forward)]
)
def test_huggingface(seq_len, model_name, forward):
    config = AutoConfig.from_pretrained(model_name)
    moe_params = hf_config_to_moe_params(config, model_name=model_name)
    input = torch.randn((seq_len, config.hidden_size), device="cuda", dtype=torch.bfloat16)

    ref_output = forward(
        config,
        input.unsqueeze(0),
        moe_params
    ).squeeze(0)    

    interleaved_glu_params = get_interleave_glu_params(input, moe_params)
    gg_interleaved_output = topk_moe_forward(
        input,
        interleaved_glu_params
    )

    assert ref_output.isfinite().all()
    assert gg_interleaved_output.isfinite().all()
    assert_close(ref_output, gg_interleaved_output)