import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.profiler as proton

#try:
# The olmoe, qwen, etc SparseMoeBlocks are basically copy-pastes of mixtral's,
# so just use the OlmoeSparseMoeBLock for the sake of keeping it simple.
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock, OlmoeMLP
HAVE_HUGGINGFACE = True
#except ImportError:
#    HAVE_HUGGINGFACE = False
    
class AnnotatedOlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        with proton.scope("routing"):
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be selected
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        with proton.scope("Expert loop"):
            for expert_idx in range(self.num_experts):
                #with proton.scope(f"Expert{expert_idx}"):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def make_huggingface_moe(
    input_dim: int,
    hidden_dim: int,
    num_experts: int,
    topk: int,
    activation: str,
    dtype: torch.dtype
):
    config = OlmoeConfig(
        hidden_size=input_dim,
        intermediate_size=hidden_dim,
        num_experts=num_experts,
        num_experts_per_tok=topk
    )
    moe = AnnotatedOlmoeSparseMoeBlock(config)
    return moe

def set_huggingface_moe_weights(moe: AnnotatedOlmoeSparseMoeBlock, router_weight, weight1, weight2):
    moe.gate.weight.data[:] = router_weight.t()
    for i in range(moe.num_experts):
        moe.experts[i].gate_proj.weight.data[:] = weight1[i, :, 0::2].t()
        moe.experts[i].up_proj.weight.data[:] = weight1[i, :, 1::2].t()
        moe.experts[i].down_proj.weight.data[:] = weight2[i].t()
    return moe