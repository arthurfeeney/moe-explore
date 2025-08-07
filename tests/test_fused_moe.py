import math
import torch
from moe_explore.triton_kernels.fused_moe import (
    fused_moe,
    FusedMoeParams
)
from moe_explore.expert_permute import expert_input_permute, expert_output_permute
from moe_explore.triton_kernels.grouped_mm_gather_scatter import get_output_rows

"""
def naive_fused_moe(
    a, 
    b,
    group_indices, 
    permute_indices,
    gather,
    scatter,
    num_tokens,
    topk,
    scales, 
):
    if gather or scatter:
        out_rows = num_tokens * topk
    else:
        out_rows = num_tokens

    out = torch.zeros((num_tokens * topk, b.size(-1)), device=a.device, dtype=a.dtype)

    for i in range(b.size(0)):
        glo, ghi = group_indices[i].item(), group_indices[i + 1].item()
        if gather:
            index = permute_indices[glo:ghi].unsqueeze(-1).expand(-1, a.size(-1)) // topk
            a_gather = torch.gather(a, dim=0, index=index)
        else:
            a_gather = a[glo:ghi]
        prod = a_gather @ b[i]
        #if scatter:
        #    index = permute_indices[glo:ghi]
        #    out[glo:ghi] = prod[index]
        #else:
        out[glo:ghi] = prod


    if scatter:
        out_size = (num_tokens, b.size(-1))
        out.mul_(scales.view(-1, 1)[permute_indices])
        output = torch.zeros(out_size, 
                             dtype=a.dtype,
                             device=a.device)
        output.scatter_reduce_(
            0,
            # expand is used because scatter_reduce_ doesn't broadcast
            permute_indices.view(-1, 1).expand(-1, out_size[-1]) // topk,
            out,
            reduce='sum'
        )
        return output

    return out
"""

def naive_grouped_mm_gather_scatter(
    a, 
    b,
    group_indices, 
    gather_indices=None,
    scatter_indices=None, 
    scales=None, 
    scales_indices=None,
    topk=None,
    output_rows=None
):
    c_rows = get_output_rows(a.size(0), gather_indices, scatter_indices, output_rows)
    c = torch.zeros(c_rows, b.size(-1), device=a.device, dtype=a.dtype)

    for i in range(b.size(0)):
        glo, ghi = group_indices[i].item(), group_indices[i + 1].item()
        if gather_indices is not None:
            index = gather_indices[glo:ghi].unsqueeze(-1).expand(-1, a.size(-1))
            a_gather = torch.gather(a, dim=0, index=index)
        else:
            a_gather = a[glo:ghi]
        prod = a_gather @ b[i]
        if scatter_indices is not None:
            if scales is not None:
                index = scales_indices[glo:ghi]
                print(index.size(), scales.size(), prod.size())
                prod *= scales[index][:, None]
            index = scatter_indices[glo:ghi].unsqueeze(-1).expand(-1, prod.size(-1))
            c.scatter_reduce_(
                    0,
                    index, 
                    prod, 
                    "sum")
        else:
            c[glo:ghi] = prod
    return c


def test_fused_moe():
    num_tokens = 10
    num_experts = 2
    K = 128
    N = 128
    topk = 1
    weight = torch.randn((num_experts, K, N), dtype=torch.bfloat16, device="cuda") / math.sqrt(N)
    group_indices = torch.tensor([0, 4, num_tokens + 1], dtype=torch.int, device="cuda")
    permute_indices = None
    scales = torch.ones((num_tokens, topk), dtype=torch.bfloat16, device="cuda")
    params = FusedMoeParams(
        weight,
        group_indices,
        permute_indices,
        False,
        False,
        num_tokens,
        topk,
        scales
    )

    input = torch.randn((num_tokens, K), dtype=torch.bfloat16, device="cuda")
    out = fused_moe(input, params)

    assert out.isfinite().all()

def random_router(num_tokens, num_experts, topk):
    scores = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16, device="cuda")
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_scores, topk_indices

def test_fused_moe2():
    num_tokens = 20
    num_experts = 4
    K = 128
    N = 128
    topk = 1
    weight = torch.randn((num_experts, K, N), dtype=torch.bfloat16, device="cuda") / math.sqrt(N)
    group_indices = torch.tensor([0, 4, 10, 15, num_tokens], dtype=torch.int, device="cuda")
    
    _, topk_indices = random_router(num_tokens, num_experts, topk)
    permute_indices = topk_indices.view(-1).argsort()
    scales = torch.rand((num_tokens, topk), dtype=torch.bfloat16, device="cuda")
    params = FusedMoeParams(
        weight,
        group_indices,
        permute_indices,
        False,
        True,
        num_tokens,
        topk,
        scales
    )

    input = torch.randn((num_tokens, K), dtype=torch.bfloat16, device="cuda")
    out = fused_moe(input, params)
    ref = naive_grouped_mm_gather_scatter(
        input,
        params.weight,
        params.group_indices,
        None,
        scatter_indices=params.permute_indices,
        topk=topk,
        scales_indices=permute_indices,
        scales=scales.view(-1)
    )

    assert out.isfinite().all()
    torch.testing.assert_close(out, ref)
