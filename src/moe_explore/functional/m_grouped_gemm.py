import torch
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm as triton_m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.k_grouped_gemm import k_grouped_gemm as triton_k_grouped_gemm, KGroupedGEMMParams
from moe_explore.functional.activation import activation as activation_func
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
    activation: Optional[str] = None,
    return_preactivation: bool = False
):
    assert tokens.dim() == 2
    assert weight.dim() == 3
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
        is_b_transposed=False,
        return_preactivation=return_preactivation
    )
    return triton_m_grouped_gemm(tokens, weight, group_indices, params)

def m_grouped_gemm_backward(
    grad_output: torch.Tensor,
    tokens: torch.Tensor,
    preactivation: Optional[torch.Tensor],
    weight: torch.Tensor,
    group_indices: torch.Tensor,
    permute_indices: torch.Tensor,
    # Did we scatter or gather in the forward pass?
    forward_gather: bool,
    forward_scatter: bool,
    num_tokens: int,
    topk: int,
    activation: Optional[str],
):
    r"""
    output = tokens * weight. This computes:
        (1) grad_tokens = grad_output * weight^T
        (2) grad_weight = tokens^T * grad_output
    The main difficulty for (2) is that both tokens and grad_output are grouped
    for the grad_weight calculation. So it currently cannot reuse the m_grouped_gemm kernel.
    """
    grad_token_params = MGroupedGEMMParams(
        # TODO: This * topk is used because inside the kernel the gather // topk...
        # Since the grad_output is the otuput of grad(scale_and_reduce), we do not want
        # to divide by topk inside the kernel. This should just be toggled inside the kernel.
        permute_indices=permute_indices * topk if forward_scatter else permute_indices,
        gather=forward_scatter,
        scatter=forward_gather,
        num_tokens=num_tokens,
        topk=topk,
        activation=None,
        is_a_transposed=False,
        is_b_transposed=True,
        pre_act_for_grad=None,
    )
    grad_tokens = triton_m_grouped_gemm(
        grad_output, 
        weight,
        group_indices, 
        grad_token_params
    ).output
    
    if forward_gather:
        # The scatter is fused, but we need to reduce across the top-k entries.
        grad_tokens = grad_tokens.view(-1, topk, tokens.size(-1)).sum(dim=1)

    grad_weight_params = KGroupedGEMMParams(
        permute_indices=permute_indices,
        gather_a=forward_gather,
        gather_b=forward_scatter,
        num_tokens=num_tokens,
        topk=topk,
        activation=None
    )
    grad_weight = triton_k_grouped_gemm(tokens, grad_output, group_indices, grad_weight_params)

    return grad_tokens, grad_weight

class MGroupedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation):
        ctx.gather, ctx.scatter, ctx.num_tokens, ctx.topk, ctx.activation = gather, scatter, num_tokens, topk, activation
        output = m_grouped_gemm_forward(tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation)
        ctx.save_for_backward(tokens, output.preactivation, weight, group_indices, permute_indices)
        return output.output
    
    @staticmethod
    def backward(ctx, grad_output):
        tokens, preactivated, weight, group_indices, permute_indices = ctx.saved_tensors
        return (
            *m_grouped_gemm_backward(
                grad_output, 
                tokens, 
                preactivated,
                weight, 
                group_indices, 
                permute_indices, 
                ctx.gather, 
                ctx.scatter, 
                ctx.num_tokens, 
                ctx.topk, 
                # TODO: activation for this should be applied before gemm kernel...
                None
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

def m_grouped_gemm(
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
    return MGroupedGEMM.apply(tokens, weight, group_indices, permute_indices, gather, scatter, num_tokens, topk, activation)

@torch.compile
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
            
    return c.to(dtype)