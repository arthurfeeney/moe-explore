import torch
from typing import List, Tuple
import triton
import triton.language as tl
from dataclasses import dataclass

@dataclass
class GroupedTokens:
    tokens: torch.nested._NestedTensor
    indices: torch.Tensor
    token_gather_indices: torch.Tensor

@torch.compile
def get_token_indices(expert_indices, num_experts_per_token):
    flat_expert_indices = expert_indices.view(-1)
    indices = flat_expert_indices.argsort()
    token_indices = indices // num_experts_per_token
    counts = flat_expert_indices.bincount()
    # torch.NestedTensor init offsets expects zero prefix
    tokens_per_expert_range = torch.empty(counts.size(0) + 1, device=counts.device, dtype=counts.dtype)
    torch.cumsum(counts, dim=0, out=tokens_per_expert_range[1:])
    tokens_per_expert_range[0] = 0
    return indices, tokens_per_expert_range, token_indices

@torch.compile
def expert_input_permute(
    tokens: torch.Tensor, 
    expert_indices: torch.Tensor, 
    num_experts_per_token: int
) -> torch.nested._NestedTensor:
    indices, tokens_per_expert_range, token_indices = get_token_indices(expert_indices, num_experts_per_token)
    gather_tokens = torch.gather(tokens, dim=0, index=token_indices.unsqueeze(1).expand(-1, tokens.size(1)))
    routed_tokens = torch.nested.nested_tensor_from_jagged(values=gather_tokens, offsets=tokens_per_expert_range)
    return GroupedTokens(
        tokens=routed_tokens,
        indices=indices,
        token_gather_indices=token_indices
    )

@torch.compile
def expert_output_permute(
    grouped_tokens: GroupedTokens,
    expert_scores: torch.Tensor,
    output_shape: Tuple[int]
) -> torch.Tensor:
    values = grouped_tokens.tokens.values()
    values.mul_(expert_scores.view(-1, 1)[grouped_tokens.indices])
    output = torch.zeros(output_shape, dtype=values.dtype, device=values.device)
    output.scatter_reduce_(
        0,
        # `scatter_reduce_` doesn't broadcast indices, so repeat indices along hidden dim. `.expand` returns a view
        grouped_tokens.token_gather_indices.view(-1, 1).expand(-1, output_shape[-1]),
        values,
        reduce='sum'
    )
    return output