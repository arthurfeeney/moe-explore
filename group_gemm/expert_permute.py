import torch
from typing import List, Tuple
import triton
import triton.language as tl
from dataclasses import dataclass

@dataclass
class GroupedTokens:
    tokens: torch.Tensor
    tokens_per_expert_range: torch.tensor
    indices: torch.Tensor
    token_gather_indices: torch.Tensor
    group_size: int

@torch.compile
def get_token_indices(expert_indices, num_experts_per_token):
    flat_expert_indices = expert_indices.view(-1)
    indices = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount()
    tokens_per_expert_range = counts.cumsum(dim=0)
    token_indices = indices // num_experts_per_token
    return indices, tokens_per_expert_range, token_indices

def expert_input_permute(
    tokens: torch.Tensor, 
    expert_indices: torch.Tensor, 
    num_experts_per_token: int
) -> GroupedTokens:
    
    indices, tokens_per_expert_range, token_indices = get_token_indices(expert_indices, num_experts_per_token)

    token_dim = tokens.size(1)
    output = torch.gather(tokens, dim=0, index=token_indices.unsqueeze(1).expand(-1, token_dim))

    return GroupedTokens(
        tokens=output,
        tokens_per_expert_range=tokens_per_expert_range,
        indices=indices, 
        token_gather_indices=token_indices,
        group_size=expert_indices.max() + 1# just pass in num_experts to kernel?
    )

def expert_output_permute(
    grouped_tokens: GroupedTokens,
    expert_scores: torch.Tensor,
    num_experts_per_token: int,
    output_shape: Tuple[int]
) -> torch.Tensor:
    grouped_tokens.tokens.mul_(expert_scores.view(-1, 1)[grouped_tokens.indices])
    output = torch.zeros(output_shape, dtype=grouped_tokens.tokens.dtype, device=grouped_tokens.tokens.device)
    output.scatter_reduce_(
        0,
        # index tensor doesn't broadcast, so repeat indices along hidden dim
        grouped_tokens.token_gather_indices.view(-1, 1).repeat(1, output_shape[-1]),
        grouped_tokens.tokens,
        reduce='sum'
    )
    
    return output