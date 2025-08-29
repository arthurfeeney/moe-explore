import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def silu(x: tl.tensor):
    return x * tl.sigmoid(x)

@triton.jit
def approx_gelu(x: tl.tensor):
    # This is the approximation of gelu used by pytorch:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    pi: tl.constexpr = 3.14159265358979323846
    tensor_cubed = x * x * x
    return 0.5 * x * (1 + libdevice.tanh(tl.sqrt(2 / pi) * (x + 0.044715 * tensor_cubed)))

@triton.jit
def gelu(x: tl.tensor):
    # The actual GELU function:
    # https://github.com/pytorch/pytorch/blob/838f22c57df8d788a55a7637f93327f5ff26cd88/torch/_refs/nn/functional/__init__.py#L1058
    SQRT_1_DIV_2: tl.constexpr = 0.70710678118654752440
    return x * 0.5 * (1 + tl.erf(x * SQRT_1_DIV_2))

TRITON_ACTIVATIONS = {
    "silu": silu,
    "approx_gelu": approx_gelu,
    "gelu": gelu
}