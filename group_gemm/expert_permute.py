import torch
from typing import List, Tuple

def get_token_indices(expert_indices, num_experts_per_token):
    flat_expert_indices = expert_indices.view(-1)
    indices = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy() # TODO: why move to cpu?
    tokens_per_expert_range = counts.cumsum()
    token_indices = indices // num_experts_per_token
    return indices, tokens_per_expert_range, token_indices

def expert_input_permute(
    tokens: torch.Tensor, 
    expert_indices: torch.Tensor, 
    num_experts_per_token: int
) -> List[torch.Tensor]:
    _, tokens_per_expert_range, token_indices = get_token_indices(expert_indices, num_experts_per_token)
    print(token_indices)
    
    expert_data = []
    for expert_id, end_idx in enumerate(tokens_per_expert_range):
        start_idx = 0 if expert_id == 0 else tokens_per_expert_range[expert_id - 1]
        if start_idx == end_idx:
            # triton group gemm needs len(group_A) == len(group_B),
            # adding an empty list is just a hack to ensure the group sizes are the
            # same. This could/should also be done by not passing in weights if the
            # corresponding expert gets no tokens.
            expert_data.append(torch.empty((0, tokens.size(-1))))
            continue
        
        exp_token_idxs = token_indices[start_idx:end_idx]
        expert_tokens = tokens[exp_token_idxs]
        expert_data.append(expert_tokens)
        
    return expert_data, tokens_per_expert_range

def expert_output_permute(
    expert_data: List[torch.Tensor], 
    expert_indices: torch.Tensor, 
    expert_weights: torch.Tensor, 
    num_experts_per_token: int,
    output_shape: Tuple[int]
) -> torch.Tensor:
    hidden_dim = expert_data[0].size(-1)
    flat_expert_weights = expert_weights.view(-1, 1)

    indices, tokens_per_expert_range, token_indices = get_token_indices(expert_indices, num_experts_per_token)
    
    output = torch.zeros(output_shape, device=expert_data[0].device, dtype=expert_data[0].dtype)
    for expert_id, end_idx in enumerate(tokens_per_expert_range):
        start_idx = 0 if expert_id == 0 else tokens_per_expert_range[expert_id - 1]
        if start_idx == end_idx:
            continue
        
        # multiply token by by expert weights and reduce
        expert_data[expert_id].mul_(flat_expert_weights[indices[start_idx:end_idx]])
        exp_token_idxs = token_indices[start_idx:end_idx]
        print('permute', expert_data[expert_id][:, :5])

        output.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).repeat(1, hidden_dim),
            expert_data[expert_id],
            reduce='sum'
        )
    
    return output
