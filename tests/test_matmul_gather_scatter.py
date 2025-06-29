import torch

from moe.matmul_gather_scatter import matmul_gather_scatter

def test_matmul_gather_scatter_group_and_gather():
    e, m, k, n = 1, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    # one group that is all of a
    group_indices = torch.tensor([0, m], device="cuda").to(torch.uint32)
    gather_indices = torch.arange(0, m, device="cuda")

    c = matmul_gather_scatter(
        a,
        b,
        group_indices,
        gather_indices,
        None,
        None
    )

    assert c.size(0) == e * a.size(0)
    ref = (a @ b).squeeze(0)
    assert torch.allclose(c, ref, atol=1e-3, rtol=1e-3)

def test_matmul_gather_scatter_group():
    e, m, k, n = 1, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)
    
    # one group that is all of a
    group_indices = torch.tensor([0, m], device="cuda").to(torch.uint32)

    c = matmul_gather_scatter(
        a,
        b,
        group_indices,
        None,
        None,
        None
    )

    assert c.size(0) == e * a.size(0)
    ref = (a @ b).squeeze(0)
    assert torch.allclose(c, ref, atol=1e-3, rtol=1e-3)

def test_matmul_gather_scatter_group_and_gather_2():
    e, m, k, n = 8, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)

    group_indices = torch.arange(0, e * m + 1, m, device="cuda")
    gather_indices = torch.arange(0, m, device="cuda").repeat(e)

    c = matmul_gather_scatter(
        a,
        b,
        group_indices,
        gather_indices,
        None,
        None
    )

    assert c.size(0) == e * a.size(0)
    assert c.size(0) == gather_indices.size(0)
    assert c.size(1) == b.size(1)
    assert torch.allclose(c[:m], a @ b[0])
    assert torch.allclose(c[m:2*m], a @ b[1])
    assert torch.allclose(c[(e-1)*m:e*m], a @ b[e-1])

def test_matmul_gather_scatter_group_and_gather_reverse():
    def reverse_rows(x):
        return torch.flip(x, dims=[0])

    e, m, k, n = 8, 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((e, k, n), device="cuda", dtype=torch.float16)

    group_indices = torch.arange(0, e * m + 1, m, device="cuda")
    # each group reverses the rows of `a`
    gather_indices = reverse_rows(torch.arange(0, m, device="cuda")).repeat(e)

    c = matmul_gather_scatter(
        a,
        b,
        group_indices,
        gather_indices,
        None,
        None
    )

    assert c.size(0) == e * a.size(0)
    assert c.size(0) == gather_indices.size(0)
    assert c.size(1) == b.size(1)
    a_rev = reverse_rows(a)
    assert torch.allclose(c[:m], a_rev @ b[0])
    assert torch.allclose(c[m:2*m], a_rev @ b[1])
    assert torch.allclose(c[(e-1)*m:e*m], a_rev @ b[e-1])

