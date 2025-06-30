from dataclasses import dataclass
import torch

from moe.matmul_gather_scatter import matmul_gather_scatter, get_output_rows

def naive_matmul_gather_scatter(a, b, group_indices, gather_indices, scatter_indices, scales, output_rows=None):

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
                prod *= scales[glo:ghi][:, None]
            index = scatter_indices[glo:ghi].unsqueeze(-1).expand(-1, prod.size(-1))
            print(c.size(), index.size(), index.min(), index.max())
            c.scatter_reduce_(
                    0,
                    index, 
                    prod, 
                    "sum")
        else:
            c[glo:ghi] = prod
    return c

def cmp(a, b, group_indices, gather_indices, scatter_indices, scales, output_rows=None):
    out = matmul_gather_scatter(a, b, group_indices, gather_indices, scatter_indices, scales)
    ref = naive_matmul_gather_scatter(a, b, group_indices, gather_indices, scatter_indices, scales, output_rows)

    assert out.dtype == a.dtype
    assert out.size() == ref.size()

    if scatter_indices is not None:
        assert out.size(0) == torch.unique(scatter_indices).size(0)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)

def test_matmul_gather_scatter_group_and_gather():
    e, m, k, n = 1, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    # one group that is all of a
    group_indices = torch.tensor([0, m], device="cuda").to(torch.uint32)
    gather_indices = torch.arange(0, m, device="cuda")

    cmp(a,
        b,
        group_indices,
        gather_indices,
        None,
        None
    )

def test_matmul_gather_scatter_group():
    e, m, k, n = 1, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    # one group that is all of a
    group_indices = torch.tensor([0, m], device="cuda").to(torch.uint32)

    cmp(a,
        b,
        group_indices,
        None,
        None,
        None
    )

def test_matmul_gather_scatter_group_sizes():
    e, m, k, n = 3, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)

    group_indices = torch.tensor([0, 14, 128, 512], device="cuda", dtype=torch.uint32) 

    cmp(a,
        b,
        group_indices,
        None,
        None,
        None
    )
    
def test_matmul_gather_scatter_group_and_gather_2():
    e, m, k, n = 8, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)

    group_indices = torch.arange(0, e * m + 1, m, device="cuda")
    gather_indices = torch.arange(0, m, device="cuda").repeat(e)

    cmp(a,
        b,
        group_indices,
        gather_indices,
        None,
        None
    )

def test_matmul_gather_scatter_group_and_gather_reverse():
    def reverse_rows(x):
        return torch.flip(x, dims=[0])

    e, m, k, n = 8, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)

    group_indices = torch.arange(0, e * m + 1, m, device="cuda")
    # each group reverses the rows of `a`
    gather_indices = reverse_rows(torch.arange(0, m, device="cuda")).repeat(e)

    cmp(a,
        b,
        group_indices,
        gather_indices,
        None,
        None
    )

def test_matmul_gather_scatter_group_scatter_1():
    e, m, k, n = 1, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    # one group that is all of a, output tokens scaled by 2 before scatter_reduce
    group_indices = torch.tensor([0, m], device="cuda").to(torch.uint32)
    scatter_indices = torch.arange(0, m, device="cuda")
    scales = 2 * torch.ones(m, device="cuda", dtype=torch.float16)

    cmp(a,
        b,
        group_indices,
        None,
        scatter_indices,
        scales
    )

def test_matmul_gather_scatter_group_scatter_1():
    e, m, k, n = 1, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    # one group that is all of a, output tokens scaled by 2 before scatter_reduce
    group_indices = torch.tensor([0, m], device="cuda").to(torch.uint32)
    gather_indices = torch.arange(0, m, device="cuda")
    scatter_indices = torch.arange(0, m, device="cuda")
    scales = 2 * torch.ones(m, device="cuda", dtype=torch.float16)

    cmp(a,
        b,
        group_indices,
        gather_indices,
        scatter_indices,
        scales
    )

def test_matmul_gather_scatter_all():
    e, m, k, n = 3, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    gather_size = 4 * m

    # four groups with random sizes
    g1 = torch.randint(low=1, high=gather_size - 2, size=(1,)).item()
    g2 = torch.randint(low=g1 + 1, high=gather_size - 1, size=(1,)).item()
    group_indices = torch.tensor([0, g1, g2, gather_size], device="cuda").to(torch.uint32)

    print(group_indices)

    # random gather and scatter indices
    gather_indices = torch.randint(0, m, size=(gather_size,), device="cuda")
    scatter_indices = torch.randint(0, m, size=(gather_size,), device="cuda")
    scales = torch.randn(gather_size, device="cuda", dtype=torch.float16)

    naive_matmul_gather_scatter(a,
        b,
        group_indices,
        gather_indices,
        scatter_indices,
        scales,
        output_rows=m 
    )
