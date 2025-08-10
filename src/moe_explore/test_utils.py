import torch
from moe_explore.router import topk_router
from typing import Optional
from moe_explore.triton_kernels.fused_moe import FusedMoeParams

def torch_grouped_matmul_gather_scatter(
    a: torch.Tensor,
    params: FusedMoeParams,
):
    r"""
    This is a reference implementation of a grouped matmul, with an optional
    fused gather / scatter-reduce operation.
    """
    dtype = a.dtype
    a = a.to(torch.float32)
    b = params.weight.to(torch.float32)
    group_indices = params.group_indices
    gather_indices = params.permute_indices // params.topk if params.gather else None
    scatter_indices = params.permute_indices if params.scatter else None
    
    if params.gather:
        c_rows = a.size(0) * params.topk
    else:
        c_rows = a.size(0)    
    c = torch.zeros(c_rows, b.size(-1), device=a.device, dtype=torch.float32)
    
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
        print(params.scales.size())
        c = (c.view(-1, params.topk, b.size(-1)) * params.scales[..., None].to(torch.float32)).sum(1)
            
    return c.to(dtype)

def random_routing(num_tokens: int, num_experts: int, topk: int, device: torch.device, dtype: torch.dtype):
    routing_logits = torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32)
    topk_logits, topk_indices = torch.topk(routing_logits, k=topk, dim=-1, sorted=False)
    return topk_logits.to(dtype), topk_indices

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