import torch
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.k_grouped_gemm import k_grouped_gemm, KGroupedGEMMParams
from typing import Optional

def m_grouped_gemm_forward(
    tokens: torch.Tensor, 
    weight: torch.Tensor, 
    group_indices: torch.Tensor,
    permute_indices: Optional[torch.Tensor],
    gather: bool,
    scatter: bool,
    num_tokens: int,
    topk: int,
    activation: Optional[str] = None
):
    assert tokens.dim() == 2
    assert weight.dim() == 3
    assert tokens.size(1) == weight.size(1)
    assert group_indices.size(0) == weight.size(0) + 1
    assert num_tokens > 0 and topk > 0
    if gather or scatter:
        assert permute_indices is not None       
    params = MGroupedGEMMParams(
        permute_indices=permute_indices,
        gather=gather,
        scatter=scatter,
        num_tokens=num_tokens,
        topk=topk,
        activation=activation,
        is_a_transposed=False,
        is_b_transposed=False
    )
    return m_grouped_gemm(tokens, weight, group_indices, params)

def m_grouped_gemm_backward(
    grad_output: torch.Tensor,
    tokens: torch.Tensor,
    weight: torch.Tensor,
    group_indices: torch.Tensor,
    permute_indices: torch.Tensor,
    gather: bool,
    scatter: bool,
    num_tokens: int,
    topk: int,
    activation: Optional[str]
):
    r"""
    output = tokens * weight. This computes:
        (1) grad_tokens = grad_output * weight^T
        (2) grad_weight = tokens^T * grad_output
    The main difficulty for (2) is that both tokens and grad_output are grouped
    for the grad_weight calculation. So it cannot reuse the batched weight kernel.
    """
    grad_params = MGroupedGEMMParams(
        permute_indices=permute_indices,
        gather=scatter,
        scatter=gather,
        num_tokens=num_tokens,
        topk=topk,
        activation=activation,
        is_a_transposed=False,
        is_b_transposed=False
    )
    grad_tokens = m_grouped_gemm(grad_output, weight.permute(0, 2, 1), group_indices, grad_params)
    
    grad_params = KGroupedGEMMParams(
        permute_indices=permute_indices,
        gather=gather,
        scatter=scatter,
        num_tokens=num_tokens,
        topk=topk,
        activation=activation,
        is_a_transposed=False,
        is_b_transposed=False
    )
    grad_weight = k_grouped_gemm(tokens, grad_output, group_indices, grad_params)
        
    return grad_tokens, grad_weight

class MGroupedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation):
        ctx.save_for_backward(tokens, weight, group_indices, permute_indices)
        ctx.gather, ctx.scatter, ctx.num_tokens, ctx.topk, ctx.activation = gather, scatter, num_tokens, topk, activation
        return m_grouped_gemm_forward(tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation)
    
    @staticmethod
    def backward(ctx, grad_output):
        tokens, weight, group_indices, permute_indices = ctx.saved_tensors
        return (
            *m_grouped_gemm_backward(
                grad_output, 
                tokens, 
                weight, 
                group_indices, 
                permute_indices, 
                ctx.gather, 
                ctx.scatter, 
                ctx.num_tokens, 
                ctx.topk, 
                ctx.activation
            ),
            None, 
            None,
            None,
            None,
            None,
            None,
            None
        )
        
grouped_gemm = MGroupedGEMM.apply

def torch_grouped_gemm(
    tokens: torch.Tensor, 
    weight: torch.Tensor, 
    group_indices: torch.Tensor,
    permute_indices: Optional[torch.Tensor],
    gather: bool,
    scatter: bool,
    num_tokens: int,
    topk: int,
    activation: Optional[str] = None
):
    dtype = tokens.dtype
    group_indices = group_indices
    gather_indices = permute_indices // topk if gather else None
    scatter_indices = permute_indices if scatter else None
    
    if gather:
        c_rows = tokens.size(0) * topk
    else:
        c_rows = tokens.size(0)    
    c = torch.zeros(c_rows, weight.size(-1), device=tokens.device, dtype=dtype)
    
    for i in range(weight.size(0)):
        glo, ghi = group_indices[i].item(), group_indices[i + 1].item()
        if gather:
            index = gather_indices[glo:ghi].unsqueeze(-1).expand(-1, tokens.size(-1))
            a_gather = torch.gather(tokens, dim=0, index=index)
        else:
            a_gather = tokens[glo:ghi]

        prod = a_gather @ weight[i]
        if scatter:
            c[scatter_indices[glo:ghi]] = prod
        else:
            c[glo:ghi] = prod
            
    if activation is not None:
        c = activation(c, activation)
            
    if scatter and scales is not None:
        c = scale_and_reduce(c, scales, num_tokens, topk, weight.size(-1))
            
    return c.to(dtype)