import torch
from typing import Optional
from moe_explore.functional.m_grouped_gemm import m_grouped_gemm_forward, m_grouped_gemm_backward, torch_grouped_gemm
from moe_explore.functional.activation import activation as activation_func
import triton.profiler as proton

def m_grouped_mlp_forward(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    group_indices: torch.Tensor,
    permute_indices: Optional[torch.Tensor],
    num_tokens: int,
    topk: int,
    activation: Optional[str] = None
):
    assert tokens.dim() == 2
    assert weight1.dim() == 3
    assert weight2.dim() == 3
    assert group_indices.size(0) == weight1.size(0) + 1
    assert group_indices.size(0) == weight2.size(0) + 1
    assert num_tokens > 0 and topk > 0
    with torch.no_grad():
        with proton.scope("mlp_forward_weight1"):
            intermediate = m_grouped_gemm_forward(
                tokens,
                weight1,
                group_indices,
                permute_indices=permute_indices,
                gather=permute_indices is not None,
                scatter=False,
                num_tokens=tokens.size(0),
                topk=topk,
                activation=activation
            )
        
        with proton.scope("mlp_forward_weight2"):
            output = m_grouped_gemm_forward(
                intermediate,
                weight2,
                group_indices,
                permute_indices=permute_indices,
                gather=False,
                scatter=permute_indices is not None,
                num_tokens=tokens.size(0),
                topk=topk,
                activation=None
            )
    
    # The intermediate state is returned for the backward pass.
    return output, intermediate

def m_grouped_mlp_backward(
    grad_output: torch.Tensor,
    tokens: torch.Tensor,
    intermediate: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    group_indices: torch.Tensor,
    permute_indices: torch.Tensor,
    num_tokens: int,
    topk: int,
    activation: Optional[str] = None
):
    assert grad_output.dim() == 2
    assert weight1.dim() == 3
    assert weight2.dim() == 3
    assert group_indices.size(0) == weight1.size(0) + 1
    assert group_indices.size(0) == weight2.size(0) + 1
    assert num_tokens > 0 and topk > 0
    with torch.no_grad():
        grad_activation: Optional[str] = "grad_" + activation if activation is not None else None
        grad_intermediate, grad_weight2 = m_grouped_gemm_backward(
            grad_output,
            intermediate,
            weight2,
            group_indices,
            permute_indices,
            forward_gather=False,
            forward_scatter=True,
            num_tokens=num_tokens,
            topk=topk,
            activation=None
        )

        # TODO: how to fuse this???
        grad_intermediate = grad_intermediate * activation_func(intermediate, grad_activation)

        grad_tokens, grad_weight1 = m_grouped_gemm_backward(
            grad_intermediate,
            tokens,
            weight1,
            group_indices,
            permute_indices,
            forward_gather=True,
            forward_scatter=False,
            num_tokens=num_tokens,
            topk=topk,
            activation=None
        )
    
    return grad_tokens, grad_weight1, grad_weight2

class MGroupedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight1, weight2, group_indices, permute_indices, num_tokens, topk, activation):
        output, intermediate = m_grouped_mlp_forward(
            tokens, weight1, weight2, group_indices, permute_indices, num_tokens, topk, activation)
        ctx.save_for_backward(tokens, intermediate, weight1, weight2, group_indices, permute_indices)
        ctx.num_tokens, ctx.topk, ctx.activation = num_tokens, topk, activation
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        tokens, intermediate, weight1, weight2, group_indices, permute_indices = ctx.saved_tensors
        return (
            *m_grouped_mlp_backward(
                grad_output, 
                tokens, 
                intermediate, 
                weight1, 
                weight2, 
                group_indices, 
                permute_indices, 
                ctx.num_tokens, 
                ctx.topk, 
                ctx.activation),
            None,
            None,
            None,
            None,
            None
        )
        
def m_grouped_mlp(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    group_indices: torch.Tensor,
    permute_indices: torch.Tensor,
    num_tokens: int,
    topk: int,
    activation: Optional[str] = None
):
    return MGroupedMLP.apply(tokens, weight1, weight2, group_indices, permute_indices, num_tokens, topk, activation)

def torch_grouped_mlp(
    tokens: torch.Tensor, 
    weight1: torch.Tensor, 
    weight2: torch.Tensor,
    group_indices: torch.Tensor,
    permute_indices: Optional[torch.Tensor],
    num_tokens: int,
    topk: int,
    activation: Optional[str] = None
):
    l1 = torch_grouped_gemm(tokens, weight1, group_indices, permute_indices, True, False, num_tokens, topk, None)
    l1 = activation_func(l1, activation)
    l2 = torch_grouped_gemm(l1, weight2, group_indices, permute_indices, False, True, num_tokens, topk, None)
    return l2