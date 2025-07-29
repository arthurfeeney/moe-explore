r"""
These are reference implementations of a few production
MoE layers. These are mostly convenient for testing.
"""

import torch
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig 
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock 

from moe_explore.params import MOEParams

class Olmoe(OlmoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)

    def init_weights(
        self,
        router_weight,
        gate_weight,
        up_weight,
        down_weight
    ):
        self.gate.weight.data = router_weight.T
        for i in range(self.num_experts):
            self.experts[i].gate_proj.weight.data = gate_weight[i].t()
            self.experts[i].up_proj.weight.data = up_weight[i].t()
            self.experts[i].down_proj.weight.data = down_weight[i].t()

def olmoe_forward(
    olmoe_config: OlmoeConfig,
    input: torch.Tensor,
    params: MOEParams
):
    moe = Olmoe(olmoe_config).to("cuda")
    moe.init_weights(
        params.expert_params.router_weight,
        params.expert_params.gate_weight,
        params.expert_params.up_weight,
        params.expert_params.down_weight
    )
    output, _ = moe(input)
    return output

class Qwen3Moe(Qwen3MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)

    def init_weights(
        self,
        router_weight,
        gate_weight,
        up_weight,
        down_weight
    ):
        self.gate.weight.data = router_weight.T
        for i in range(self.num_experts):
            self.experts[i].gate_proj.weight.data = gate_weight[i].t()
            self.experts[i].up_proj.weight.data = up_weight[i].t()
            self.experts[i].down_proj.weight.data = down_weight[i].t()

def qwen3_moe_forward(
    qwen3_config: Qwen3MoeConfig,
    input: torch.Tensor,
    params: MOEParams
):
    moe = Qwen3Moe(qwen3_config).to("cuda")
    moe.init_weights(
        params.expert_params.router_weight,
        params.expert_params.gate_weight,
        params.expert_params.up_weight,
        params.expert_params.down_weight
    )
    output, _ = moe(input)
    return output
