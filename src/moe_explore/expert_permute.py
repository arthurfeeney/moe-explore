import torch
from typing import Union, Tuple
from dataclasses import dataclass

@dataclass
class GroupedTokens:
    tokens: torch.Tensor
    group_indices: torch.Tensor
    indices: torch.Tensor

@dataclass
class PermToGroupIndices:
    group_indices: torch.Tensor
    indices: torch.Tensor

@torch.compile
def get_token_indices(
    expert_indices, 
    topk, 
    num_experts,
    zero_prefix=False
):
    flat_expert_indices = expert_indices.view(-1)
    indices = flat_expert_indices.argsort()#.to(torch.int32)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=expert_indices.device)
    torch.histc(flat_expert_indices, min=0, max=num_experts - 1, bins=num_experts, out=counts)

    if zero_prefix:
        group_indices = torch.empty(counts.size(0) + 1, dtype=torch.int32, device=expert_indices.device)
        group_indices[0] = 0
        torch.cumsum(counts, dim=0, out=group_indices[1:])
    else:
        group_indices = counts.cumsum(dim=0)
    
    return PermToGroupIndices(
        group_indices=group_indices,
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
    output = torch.gather(tokens, dim=0, index=indices.indices.unsqueeze(1).expand(-1, token_dim) // topk)
    return GroupedTokens(
        tokens=output,
        group_indices=indices.group_indices,
        indices=indices.indices 
    )

@torch.compile
def expert_output_permute(
    grouped_tokens: GroupedTokens,
    expert_scores: torch.Tensor,
    topk: int,
    output_shape: Union[Tuple[int], torch.Size]
) -> torch.Tensor:
    grouped_tokens.tokens.mul_(expert_scores.view(-1, 1)[grouped_tokens.indices])
    output = torch.zeros(output_shape, dtype=grouped_tokens.tokens.dtype, device=grouped_tokens.tokens.device)
    output.scatter_reduce_(
        0,
        # expand is used because scatter_reduce_ doesn't broadcast
        grouped_tokens.indices.view(-1, 1).expand(-1, output_shape[-1]) // topk,
        grouped_tokens.tokens,
        reduce='sum'
    )
    return output