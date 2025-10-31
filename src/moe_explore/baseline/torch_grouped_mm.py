import torch
import triton.profiler as proton
from moe_explore.expert_permute import expert_input_permute, expert_output_permute
from moe_explore.functional.activation import activation as activation_func

class TorchGroupedMMMoE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        topk: int,
        activation: str,
        dtype
    ):
        super().__init__()
        
        self.activation = activation
        self.topk = topk
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        # currently only implement GLUs
        assert "glu" in activation
        if "glu" in activation:
            self.weight1 = torch.nn.Parameter(torch.empty(self.num_experts, input_dim, hidden_dim, dtype=dtype))
            self.weight2 = torch.nn.Parameter(torch.empty(self.num_experts, input_dim, hidden_dim, dtype=dtype))
            self.weight3 = torch.nn.Parameter(torch.empty(self.num_experts, hidden_dim, input_dim, dtype=dtype))
                    
    def init_weights(self, weight1, weight2):
        self.weight1.data[:] = weight1[..., 0::2]
        self.weight2.data[:] = weight1[..., 1::2]
        self.weight3.data[:] = weight2[:]
        
    def forward(
        self,
        tokens: torch.Tensor,
        router,
    ):
        topk_scores, topk_indices, router_logits = router(tokens)
        grouped_tokens = expert_input_permute(
            tokens,
            topk_indices,
            self.num_experts,
            self.topk,
            zero_prefix=False
        )
        gate = torch._grouped_mm(grouped_tokens.tokens, self.weight1, grouped_tokens.group_indices)    
        up = torch._grouped_mm(grouped_tokens.tokens, self.weight2, grouped_tokens.group_indices)
        output = activation_func((gate, up), self.activation)
        grouped_tokens.tokens = torch._grouped_mm(output, self.weight3, grouped_tokens.group_indices)
        output = expert_output_permute(grouped_tokens, topk_scores, self.topk, tokens.shape)
        return output, router_logits