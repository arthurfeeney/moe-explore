import torch
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
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
        activation=activation
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
        1. grad_tokens = grad_output * weight^T
        2. grad_weight = tokens^T * grad_output
    The main difficulty is that both tokens and grad_output are grouped
    for the grad_weight calculation.
    """
    grad_tokens_params = MGroupedGEMMParams(
        permute_indices=permute_indices,
        # We need to do the opposite of gather/scatter done in the forward pass.
        gather=scatter,
        scatter=gather,
        num_tokens=num_tokens,
        topk=topk,
    )
    grad_tokens = m_grouped_gemm(grad_output, weight.permute(0, 2, 1), group_indices, grad_tokens_params)
    grad_weight = torch.zeros_like(weight)
    return grad_tokens, grad_weight



