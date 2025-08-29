from functools import partial
import pytest
import torch
import triton
import triton.language as tl
from moe_explore.triton_kernels.activation import silu, gelu, approx_gelu

@triton.jit
def activation_kernel(x_ptr, y_ptr, ACTIVATION: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(x_ptr + offsets)
    y = ACTIVATION(x)
    tl.store(y_ptr + offsets, y)

@pytest.mark.parametrize("triton_act, torch_act, atol, rtol", [
    (silu, torch.nn.functional.silu, 1e-6, 1e-6),
    (gelu, torch.nn.functional.gelu, 1e-6, 1e-6),
    (approx_gelu, partial(torch.nn.functional.gelu, approximate="tanh"), 1e-6, 1e-6),
])
def test_activation(triton_act, torch_act, atol, rtol):
    x = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    y_ref = torch_act(x)
    y = torch.empty_like(x)
    activation_kernel[(1, 1)](x, y, triton_act, x.size(0), x.size(1))
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)