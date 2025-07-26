import torch
from typing import Union, Tuple
from dataclasses import dataclass

@dataclass
class GroupedTokens:
    tokens: torch.Tensor
    group_indices: torch.Tensor
    permute_indices: torch.Tensor
    indices: torch.Tensor

@dataclass
class PermToGroupIndices:
    group_indices: torch.Tensor
    permute_indices: torch.Tensor
    indices: torch.Tensor

@torch.compile
def get_token_indices(
    expert_indices, 
    topk, 
    num_experts,
    zero_prefix=False
):
    flat_expert_indices = expert_indices.view(-1)
    indices = flat_expert_indices.argsort()
    counts = torch.histc(flat_expert_indices, min=0, max=num_experts - 1, bins=num_experts)

    if zero_prefix:
        group_indices = torch.empty(counts.size(0) + 1, dtype=torch.int64, device=expert_indices.device)
        group_indices[0] = 0
        torch.cumsum(counts, dim=0, out=group_indices[1:])
    else:
        group_indices = counts.cumsum(dim=0)

    permute_indices = indices // topk
    return PermToGroupIndices(
        group_indices=group_indices,
        permute_indices=permute_indices,
        indices=indices
    )

@torch.compile
def expert_input_permute(
    tokens: torch.Tensor, 
    expert_indices: torch.Tensor, 
    num_experts: int,
    topk: int
) -> GroupedTokens:
    indices = get_token_indices(expert_indices, topk, num_experts, zero_prefix=True)
    token_dim = tokens.size(1)
    output = torch.gather(tokens, dim=0, index=indices.permute_indices.unsqueeze(1).expand(-1, token_dim))
    return GroupedTokens(
        tokens=output,
        group_indices=indices.group_indices,
        permute_indices=indices.permute_indices,
        indices=indices.indices 
    )

@torch.compile
def expert_output_permute(
    grouped_tokens: GroupedTokens,
    expert_scores: torch.Tensor,
    output_shape: Union[Tuple[int], torch.Size]
) -> torch.Tensor:
    grouped_tokens.tokens.mul_(expert_scores.view(-1, 1)[grouped_tokens.indices])
    output = torch.zeros(output_shape, dtype=grouped_tokens.tokens.dtype, device=grouped_tokens.tokens.device)
    output.scatter_reduce_(
        0,
        # expand is used because scatter_reduce_ doesn't broadcast
        grouped_tokens.permute_indices.view(-1, 1).expand(-1, output_shape[-1]),
        grouped_tokens.tokens,
        reduce='sum'
    )
    return output
