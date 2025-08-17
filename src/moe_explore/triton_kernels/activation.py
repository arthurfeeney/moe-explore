import triton
import triton.language as tl

@triton.jit
def silu(tensor):
    return tensor * tl.sigmoid(tensor)

@triton.jit
def approx_gelu(tensor):
    # This is the approximation of gelu used by pytorch:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    pi: tl.constexpr = 3.14159265358979323846
    tensor_cubed = tensor * tensor * tensor
    return 0.5 * tensor * (1 + tl.extra.device.tanh(tl.sqrt(2 / pi) * (tensor + 0.044715 * tensor_cubed)))

@triton.jit
def gelu(tensor):
    # The actual GELU function:
    # https://github.com/pytorch/pytorch/blob/838f22c57df8d788a55a7637f93327f5ff26cd88/torch/_refs/nn/functional/__init__.py#L1058
    SQRT_1_DIV_2: tl.constexpr = 0.70710678118654752440
    return tensor * 0.5 * (1 + tl.erf(tensor * SQRT_1_DIV_2))

TRITON_ACTIVATIONS = {
    "silu": silu,
    "approx_gelu": approx_gelu,
    "gelu": gelu
}