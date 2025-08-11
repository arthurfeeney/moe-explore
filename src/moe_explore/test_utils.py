import math
import torch
import torch.nn.functional as F
from moe_explore.triton_kernels.fused_moe import FusedMoeParams
from moe_explore.params import MLPParams, GLUParams

def torch_grouped_matmul_gather_scatter(
    a: torch.Tensor,
    params: FusedMoeParams,
):
    r"""
    This is a reference implementation of a grouped matmul, with an optional
    fused gather / scatter-reduce operation.
    """
    dtype = a.dtype
    a = a
    b = params.weight
    group_indices = params.group_indices
    gather_indices = params.permute_indices // params.topk if params.gather else None
    scatter_indices = params.permute_indices if params.scatter else None
    
    if params.gather:
        c_rows = a.size(0) * params.topk
    else:
        c_rows = a.size(0)    
    c = torch.zeros(c_rows, b.size(-1), device=a.device, dtype=dtype)
    
    for i in range(b.size(0)):
        glo, ghi = group_indices[i].item(), group_indices[i + 1].item()
        if params.gather:
            index = gather_indices[glo:ghi].unsqueeze(-1).expand(-1, a.size(-1))
            a_gather = torch.gather(a, dim=0, index=index)
        else:
            a_gather = a[glo:ghi]
            
        prod = a_gather @ b[i]
        if params.scatter:
            c[scatter_indices[glo:ghi]] = prod
        else:
            c[glo:ghi] = prod
            
    if params.scatter and params.scales is not None:
        c = (c.view(-1, params.topk, b.size(-1)) * params.scales[..., None]).sum(1)
            
    return c

def random_routing(num_tokens: int, num_experts: int, topk: int, device: torch.device, dtype: torch.dtype):
    r"""
    This generates a random routing with the routing logies selected from a normal distribution.
    This tends to generate a very balanced routing.
    """
    scores = F.softmax(torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32), dim=-1)
    topk_logits, topk_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_logits.to(dtype), topk_indices

def random_skewed_routing(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    num_skewed_experts: int,
    skew_factor: int, 
    device: torch.device, 
    dtype: torch.dtype
):
    r"""
    This generates a random routing based on a sort of simple skewed distribution.
    This makes it so that a subset of `num_skewed_experts` experts are `skew_factor` times
    more likely to be routed to than the rest.
    """
    skewed_experts_indices = torch.randperm(num_experts)[:num_skewed_experts]
    probabilities = torch.ones(num_experts, device=device, dtype=torch.float32)
    probabilities[skewed_experts_indices] *= skew_factor
    topk_indices = torch.multinomial(probabilities, num_samples=topk, replacement=False)
    # Generate random scores. Applying softmax, even though it's nonsense numbers,
    # so the distribution looks like a normal MoE router output.
    topk_logits = torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32)
    topk_scores = F.softmax(topk_logits, dim=-1)
    return topk_scores.to(dtype), topk_indices

def random_groups(num_tokens, num_groups, device: torch.device):
    r"""
    Makes a `group_inidces` tensor like [0, r1, r2, ..., rN, num_tokens]. 
    The ri are random indices in [1, num_tokens - 1].
    """
    indices = torch.arange(1, num_tokens - 1, device=device)
    perm = torch.randperm(num_tokens - 2, device=device)
    inner_group_indices, _ = torch.sort(indices[perm[:num_groups - 1]])
    group_indices = torch.empty(num_groups + 1, device=device, dtype=torch.int32)
    group_indices[0] = 0
    group_indices[1:-1] = inner_group_indices
    group_indices[-1] = num_tokens
    return group_indices

def random_mlp(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=torch.randn,
):
    return MLPParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype) / math.sqrt(num_experts),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim),
        activation
    )

def random_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=torch.randn,
):
    return GLUParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype) / math.sqrt(num_experts),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim),
        activation
    )
