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
    
    # always fuse the activation during inference.
    # If fusing during training, we must return the preactivation.
    in_train = weight1.requires_grad
    fused_act = True
    # If activation is a relu, it's gradient can be computed from the grad output.
    return_pre_act = in_train and activation != "relu"
    
    out1 = m_grouped_gemm_forward(
        tokens,
        weight1,
        group_indices,
        permute_indices=permute_indices,
        gather=permute_indices is not None,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        activation=activation if fused_act else None,
        return_preactivation=return_pre_act and fused_act
    )
    
    # If we did not fuse the activation, gemm.output is the pre-activation.
    # If we're training, need to save the pre-activation
    if activation is not None and not fused_act:
        activated = activation_func(out1.output, activation)
        preactivation = out1.output if return_pre_act else None
    else:
        activated = out1.output
        preactivation = out1.preactivation if return_pre_act else None
        
    out2 = m_grouped_gemm_forward(
        activated,
        weight2,
        group_indices,
        permute_indices=permute_indices,
        gather=False,
        scatter=permute_indices is not None,
        num_tokens=num_tokens,
        topk=topk,
        activation=None
    )

    return out2.output, activated, preactivation

def m_grouped_mlp_backward(
    grad_output: torch.Tensor,
    tokens: torch.Tensor,
    intermediate: torch.Tensor,
    pre_activation: torch.Tensor,
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
    
    grad_activation: Optional[str] = "grad_" + activation if activation is not None else None
    grad_intermediate, grad_weight2 = m_grouped_gemm_backward(
        grad_output,
        intermediate,
        None,
        weight2,
        group_indices,
        permute_indices,
        forward_gather=False,
        forward_scatter=True,
        num_tokens=num_tokens,
        topk=topk,
        activation=None
    )

    # Maybe better to fuse with the next gemm
    if activation is not None:
        # ReLU doesn't need to store pre_activation.
        if grad_activation == "grad_relu":
            grad_intermediate = activation_func(intermediate, grad_activation, grad_intermediate)
        else:
            grad_intermediate = activation_func(pre_activation, grad_activation, grad_intermediate)

    grad_tokens, grad_weight1 = m_grouped_gemm_backward(
        grad_intermediate,
        tokens,
        None,
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
        output, intermediate, preactivation = m_grouped_mlp_forward(
            tokens, weight1, weight2, group_indices, permute_indices, num_tokens, topk, activation)
        ctx.save_for_backward(tokens, intermediate, preactivation, weight1, weight2, group_indices, permute_indices)
        ctx.num_tokens, ctx.topk, ctx.activation = num_tokens, topk, activation
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        tokens, intermediate, pre_activation, weight1, weight2, group_indices, permute_indices = ctx.saved_tensors
        return (
            *m_grouped_mlp_backward(
                grad_output, 
                tokens, 
                intermediate, 
                pre_activation,
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