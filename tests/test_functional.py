from dataclasses import dataclass
import pytest
import torch
import math
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from typing import Callable, Union

from moe_explore.functional.mlp import (
    moe_mlp_torch, 
    moe_mlp_grouped_gemm_fused,
    moe_mlp_grouped_gemm
)
from moe_explore.functional.glu import (
    moe_glu_torch,
    moe_glu_grouped_gemm_fused,
    moe_glu_grouped_gemm
)
from moe_explore.hf_moe_reference import olmoe_forward

@dataclass
class MLPInput:
    input: torch.Tensor
    router_weight: torch.Tensor
    weight1: torch.Tensor
    weight2: torch.Tensor

@dataclass
class GLUInput:
    input: torch.Tensor
    router_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor

@dataclass
class Params:
    device: Union[torch.device, str]
    dtype: torch.dtype
    seq_len: int
    input_dim: int
    hidden_dim: int
    num_experts: int
    topk: int
    activation: Callable[[torch.Tensor], torch.Tensor]

    def random(self, size, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return torch.randn(size, device=self.device, dtype=dtype)

    def generate_mlp_inputs(self):
        input = self.random((self.seq_len, self.input_dim)) #torch.randn((self.seq_len, self.input_dim), device=device, dtype=dtype)
        router_weight = self.random((self.input_dim, self.num_experts)) / math.sqrt(self.num_experts)
        weight1 = self.random((self.num_experts, self.input_dim, self.hidden_dim)) / math.sqrt(self.input_dim)
        weight2 = self.random((self.num_experts, self.hidden_dim, self.input_dim)) / math.sqrt(self.hidden_dim)
        return MLPInput(input, router_weight, weight1, weight2)

    def generate_glu_inputs(self):
        input = self.random((self.seq_len, self.input_dim)) #torch.randn((self.seq_len, self.input_dim), device=device, dtype=dtype)
        router_weight = self.random((self.input_dim, self.num_experts)) / math.sqrt(self.num_experts)
        gate = self.random((self.num_experts, self.input_dim, self.hidden_dim)) / math.sqrt(self.input_dim)
        up = self.random((self.num_experts, self.input_dim, self.hidden_dim)) / math.sqrt(self.hidden_dim)
        down = self.random((self.num_experts, self.hidden_dim, self.input_dim)) / math.sqrt(self.hidden_dim)
        return GLUInput(input, router_weight, gate_weight=gate, up_weight=up, down_weight=down)

test_params = [
    Params('cuda', torch.float16, 128, 128, 256, num_experts=8, topk=2, activation=torch.nn.functional.gelu),
    Params('cuda', torch.float16, 256, 1024, 1024, 8, 2, torch.nn.functional.gelu),
    Params('cuda', torch.float16, 999, 2048, 2048, 8, 2, torch.nn.functional.gelu),
    # As of triton 3.3.1, atomic_add does not supprot bfloat16,
    # leaving commented as I believe it's supported in newer versions
    #Params('cuda', torch.bfloat16, 5, 2048, 2048, 8, 2, torch.nn.functional.gelu),
    Params('cuda', torch.float32, 257, 2048, 2048, 8, 2, torch.nn.functional.gelu)
]

@pytest.mark.parametrize("params", test_params)
def test_moe_mlp_torch(
        params
):
    inputs = params.generate_mlp_inputs()

    output = moe_mlp_torch(
        inputs.input,
        inputs.router_weight,
        inputs.weight1,
        inputs.weight2,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    assert output.size() == inputs.input.size()

@pytest.mark.parametrize("params", test_params)
def test_moe_mlp_grouped_gemm(
    params
):
    inputs = params.generate_mlp_inputs()

    gg_output = moe_mlp_grouped_gemm(
        inputs.input,
        inputs.router_weight,
        inputs.weight1,
        inputs.weight2,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    ref_output = moe_mlp_torch(
        inputs.input,
        inputs.router_weight,
        inputs.weight1,
        inputs.weight2,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    assert gg_output.size() == inputs.input.size()
    assert gg_output.size() == ref_output.size()
    assert torch.allclose(ref_output, gg_output, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("params", test_params)
def test_moe_mlp_grouped_gemm_fused(
    params
):
    inputs = params.generate_mlp_inputs()

    gg_output = moe_mlp_grouped_gemm_fused(
        inputs.input,
        inputs.router_weight,
        inputs.weight1,
        inputs.weight2,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    ref_output = moe_mlp_torch(
        inputs.input,
        inputs.router_weight,
        inputs.weight1,
        inputs.weight2,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    assert gg_output.size() == inputs.input.size()
    assert gg_output.size() == ref_output.size()
    assert torch.allclose(ref_output, gg_output, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("params", test_params)
def test_moe_glu_grouped_gemm_fused(
    params
):
    inputs = params.generate_glu_inputs()

    gg_fused_output = moe_glu_grouped_gemm_fused(
        inputs.input,
        inputs.router_weight,
        inputs.gate_weight,
        inputs.up_weight,
        inputs.down_weight,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    gg_output = moe_glu_grouped_gemm(
        inputs.input,
        inputs.router_weight,
        inputs.gate_weight,
        inputs.up_weight,
        inputs.down_weight,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    ref_output = moe_glu_torch(
        inputs.input,
        inputs.router_weight,
        inputs.gate_weight,
        inputs.up_weight,
        inputs.down_weight,
        params.input_dim,
        params.num_experts,
        params.topk,
        params.activation
    )

    assert gg_output.size() == inputs.input.size()
    assert gg_output.size() == ref_output.size()
    assert torch.allclose(ref_output, gg_output, atol=1e-1, rtol=4e-2)
    assert torch.allclose(ref_output, gg_fused_output, atol=1e-1, rtol=4e-2)

@pytest.mark.parametrize(
    "device,seq_len,input_dim,hidden_dim",
    [
        ("cuda", 128, 2048, 1024)
    ])
def test_olmoe(device, seq_len, input_dim, hidden_dim):
    config = OlmoeConfig(
       hidden_size=input_dim,
       intermediate_size=hidden_dim,
    )

    params = Params(
        device,
        torch.bfloat16,
        seq_len,
        input_dim,
        hidden_dim,
        config.num_experts,
        config.num_experts_per_tok,
        torch.nn.functional.silu
    )

    glu_input: GLUInput = params.generate_glu_inputs()

    olmoe_output = olmoe_forward(
        config,
        glu_input.input.unsqueeze(0),
        glu_input.router_weight,
        glu_input.gate_weight,
        glu_input.up_weight,
        glu_input.down_weight
    )    

    gg_output = moe_glu_grouped_gemm(
        glu_input.input,
        glu_input.router_weight,
        glu_input.gate_weight,
        glu_input.up_weight,
        glu_input.down_weight,
        params.input_dim,
        params.num_experts,
        params.topk,
        activation=torch.nn.functional.silu
    )

    assert torch.allclose(olmoe_output, gg_output, atol=1, rtol=1e-4)
