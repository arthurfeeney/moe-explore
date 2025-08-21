r"""
These are reference implementations of a few production
MoE layers. These are mostly convenient for testing.
"""

import torch
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig 
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock 
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig 
from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeSparseMoeBlock 

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
        params.router_params.router_weight,
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
        params.router_params.router_weight,
        params.expert_params.gate_weight,
        params.expert_params.up_weight,
        params.expert_params.down_weight
    )
    output, _ = moe(input)
    return output

class Ernie4_5_Moe(Ernie4_5_MoeSparseMoeBlock):
    def __init__(self, config):
        # NOTE: explicitly disabling shared experts for testing
        config.moe_num_shared_experts = 0
        super().__init__(config)

    def init_weights(
        self,
        router_weight,
        router_bias,
        gate_weight,
        up_weight,
        down_weight
    ):
        self.gate.weight.data = router_weight.T
        self.moe_statics.e_score_correction_bias.data = router_bias.unsqueeze(0)
        for i in range(self.num_experts):
            self.experts[i].gate_proj.weight.data = gate_weight[i].t()
            self.experts[i].up_proj.weight.data = up_weight[i].t()
            self.experts[i].down_proj.weight.data = down_weight[i].t()

def ernie4_5_moe_forward(
    ernie4_5_config: Ernie4_5_MoeConfig,
    input: torch.Tensor,
    params: MOEParams
):
    moe = Ernie4_5_Moe(ernie4_5_config).to("cuda")
    moe.init_weights(
        params.router_params.router_weight.to(torch.float32),
        params.router_params.bias.to(torch.float32),
        params.expert_params.gate_weight,
        params.expert_params.up_weight,
        params.expert_params.down_weight)
    output, _ = moe(input)
    return output