import torch

try:
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
    from transformer_engine.pytorch.permutation import (
        moe_permute,
        moe_unpermute
    )
    HAVE_TRANSFORMER_ENGINE = True
except ImportError:
    HAVE_TRANSFORMER_ENGINE = False
    print("failed to import transformer-engine")

from moe_explore.functional.activation import activation as activation_func

class TransformerEngineMoE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        topk: int,
        activation: str,
        dtype
    ):
        assert HAVE_TRANSFORMER_ENGINE
        super().__init__()
        self.gl1 = GroupedLinear(
            num_experts,
            input_dim,
            hidden_dim * 2 if "glu" in activation else hidden_dim,
            bias=False,
            params_dtype=dtype
        )
        self.gl2 = GroupedLinear(
            num_experts,
            hidden_dim,
            input_dim,
            bias=False,
            params_dtype=dtype
        )
        self.activation = activation
        self.topk = topk
        self.num_experts = num_experts
        self.input_dim = input_dim

    def init_weights(self, weight1, weight2):
        # This is just to test against other implementations
        for i in range(self.num_experts):
            getattr(self.gl1, f"weight{i}").data[:] = weight1[i].t()
            getattr(self.gl2, f"weight{i}").data[:] = weight2[i].t()

    def forward(
        self,
        tokens: torch.Tensor,
        router,
    ):
        assert tokens.dim() == 2
        topk_scores, topk_indices, _ = router(tokens)
        permuted_tokens, row_id_map = moe_permute(
            tokens,
            topk_indices.to(torch.int32),
            # Since we aren't doing distributed, don't
            # need to worry about capacity or OOM due to routing inbalance
            num_out_tokens=tokens.size(0) * self.topk,
            max_token_num=tokens.size(0),
            map_type="index"
        )
        m_splits = torch.histc(topk_indices.view(-1), min=0, max=self.num_experts - 1, bins=self.num_experts).tolist()
        output = self.gl1(permuted_tokens, m_splits=m_splits, is_first_microbatch=None)
        output = activation_func(output, self.activation)
        output = self.gl2(output, m_splits=m_splits, is_first_microbatch=None)
        output = moe_unpermute(output, row_id_map, map_type="index")
        # unpermute lays out like [topk, num_tokens, dim], so transpose the topk_scores
        output = (output.view(self.topk, tokens.size(0), -1) * topk_scores.permute(1, 0)[..., None]).sum(dim=0)
        return output