import torch
from typing import Union, Tuple
from dataclasses import dataclass

@dataclass
class GroupedTokens:
    tokens: torch.Tensor
    group_indices: torch.Tensor
    permute_indices: torch.Tensor
    indices: torch.Tensor

@torch.compile
def get_token_indices(expert_indices, topk, zero_prefix=False):
    flat_expert_indices = expert_indices.view(-1)
    # TODO: is the argsort really necessary?
    indices = flat_expert_indices.argsort()
    # TODO: bincount is slow and not compileable.
    # can be replaced with histc
    counts = flat_expert_indices.bincount()

    if zero_prefix:
        group_indices = torch.empty(counts.size(0) + 1, dtype=torch.int64, device=expert_indices.device)
        group_indices[0] = 0
        torch.cumsum(counts, dim=0, out=group_indices[1:])
    else:
        group_indices = counts.cumsum(dim=0)
    permute_indices = indices // topk
    return indices, group_indices, permute_indices

def expert_input_permute(
    tokens: torch.Tensor, 
    expert_indices: torch.Tensor, 
    topk: int
) -> GroupedTokens:
    indices, group_indices, permute_indices = get_token_indices(expert_indices, topk, zero_prefix=True)
    token_dim = tokens.size(1)
    output = torch.gather(tokens, dim=0, index=permute_indices.unsqueeze(1).expand(-1, token_dim))
    return GroupedTokens(
        tokens=output,
        group_indices=group_indices,
        permute_indices=permute_indices,
        indices=indices, 
    )

def expert_output_permute(
    grouped_tokens: GroupedTokens,
    expert_scores: torch.Tensor,
    output_shape: Union[Tuple[int], torch.Size]
) -> torch.Tensor:
    grouped_tokens.tokens.mul_(expert_scores.view(-1, 1)[grouped_tokens.indices])
    output = torch.zeros(output_shape, dtype=grouped_tokens.tokens.dtype, device=grouped_tokens.tokens.device)
    output.scatter_reduce_(
        0,
        # scatter_reduce_ doesn't broadcast indices, so repeat indices along hidden dim. `.expand` returns a view
        grouped_tokens.permute_indices.view(-1, 1).expand(-1, output_shape[-1]),
        grouped_tokens.tokens,
        reduce='sum'
    )
    return output
