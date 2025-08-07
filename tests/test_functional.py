from dataclasses import dataclass
import pytest
import torch
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
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
from moe_explore.params import MOEParams, random_glu, random_mlp
from moe_explore.hf_moe_reference import qwen3_moe_forward, olmoe_forward

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
        denom = size[-1]
        return torch.randn(size, device=self.device, dtype=dtype) / denom

    def random_input(self):
        return self.random((self.seq_len, self.input_dim))

    def random_params(self, func):
        return func(
            self.num_experts,
            self.input_dim,
            self.hidden_dim,
            self.activation,
            self.device,
            self.dtype
        )

    def generate_mlp_inputs(self):
        return MOEParams(
            self.random_params(random_mlp),
            self.num_experts,
            self.topk
        )

    def generate_glu_inputs(self):
        return MOEParams(
            self.random_params(random_glu),
            self.num_experts,
            self.topk
        )

test_params = [
    Params('cuda', torch.float16, 128, 128, 256, num_experts=8, topk=2, activation=torch.nn.functional.gelu),
    Params('cuda', torch.float16, 256, 1024, 1024, 8, 2, torch.nn.functional.gelu),
    Params('cuda', torch.float16, 999, 2048, 2048, 8, 2, torch.nn.functional.gelu),
    # As of triton 3.3.1, atomic_add does not supprot bfloat16,
    # there isn't an instrction for it until sm90+.
    # leaving commented as I believe it's supported in newer versions
    # Params('cuda', torch.bfloat16, 5, 2048, 2048, 8, 2, torch.nn.functional.gelu),
    Params('cuda', torch.float32, 257, 2048, 2048, 8, 2, torch.nn.functional.gelu)
]

@pytest.mark.parametrize("params", test_params)
def test_moe_mlp_torch(
        params
):
    input = params.random_input()
    moe_params = params.generate_mlp_inputs()

    output = moe_mlp_torch(
        input,
        moe_params
    )

    assert output.size() == input.size()

@pytest.mark.parametrize("params", test_params)
def test_moe_mlp_grouped_gemm(
    params
):
    input = params.random_input()
    moe_params = params.generate_mlp_inputs()

    gg_fused_output = moe_mlp_grouped_gemm_fused(
        input,
        moe_params
    )

    gg_output = moe_mlp_grouped_gemm(
        input,
        moe_params
    )

    ref_output = moe_mlp_torch(
        input,
        moe_params
    )

    assert gg_output.isfinite().all()
    assert gg_fused_output.isfinite().all()
    torch.testing.assert_close(ref_output, gg_output)
    torch.testing.assert_close(ref_output, gg_fused_output)

@pytest.mark.parametrize("params", test_params)
def test_moe_glu_grouped_gemm_fused(
    params
):
    input = params.random_input()
    moe_params = params.generate_glu_inputs()

    gg_fused_output = moe_glu_grouped_gemm_fused(
        input,
        moe_params
    )

    gg_output = moe_glu_grouped_gemm(
        input,
        moe_params
    )

    ref_output = moe_glu_torch(
        input,
        moe_params
    )

    assert gg_output.isfinite().all()
    assert gg_fused_output.isfinite().all()
    torch.testing.assert_close(ref_output, gg_output)
    torch.testing.assert_close(ref_output, gg_fused_output)

@pytest.mark.parametrize(
    "device,seq_len,input_dim,hidden_dim,Config,forward",
    [
        ("cuda", 128, 2048, 1024, OlmoeConfig, olmoe_forward),
        ("cuda", 128, 1024, 768, Qwen3MoeConfig, qwen3_moe_forward)
    ])
def test_olmoe(device, seq_len, input_dim, hidden_dim, Config, forward):

    config = Config(
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

    input = params.random_input()
    moe_params = params.generate_glu_inputs()

    ref_output = forward(
        config,
        input.unsqueeze(0),
        moe_params
    ).squeeze(0)    

    gg_output = moe_glu_grouped_gemm(
        input,
        moe_params
    )

    gg_fused_output = moe_glu_grouped_gemm_fused(
        input,
        moe_params
    )

    assert ref_output.isfinite().all()
    assert gg_output.isfinite().all()
    assert gg_fused_output.isfinite().all()
    torch.testing.assert_close(ref_output, gg_output)
    torch.testing.assert_close(ref_output, gg_fused_output)
