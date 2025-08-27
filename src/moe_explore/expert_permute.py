import torch
from typing import Union, Tuple
from dataclasses import dataclass
import triton.profiler as proton
from moe_explore.triton_kernels.row_gather_scatter import row_gather, row_scatter

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
    indices = flat_expert_indices.argsort()
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

#@torch.compile(dynamic=True)
def expert_input_permute(
    tokens: torch.Tensor, 
    expert_indices: torch.Tensor, 
    num_experts: int,
    topk: int
) -> GroupedTokens:
    indices = get_token_indices(expert_indices, topk, num_experts, zero_prefix=True)
    token_dim = tokens.size(1)
    output = row_gather(tokens, indices.indices // topk)
    return GroupedTokens(
        tokens=output,
        group_indices=indices.group_indices,
        indices=indices.indices 
    )

# TODO: This seems faster without torch.compile
@torch.compiler.disable()
def expert_output_permute(
    grouped_tokens: GroupedTokens,
    expert_scores: torch.Tensor,
    topk: int,
    output_shape: Union[Tuple[int], torch.Size]
) -> torch.Tensor:
    with proton.scope("scatter"):
        tokens = row_scatter(grouped_tokens.tokens, grouped_tokens.indices)
    with proton.scope("scale-and-reduce"):
        return torch.einsum("tkd,tk->td", tokens.view(-1, topk, tokens.size(1)), expert_scores)