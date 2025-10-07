import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def relu(x: tl.tensor):
    return tl.maximum(x, 0)

@triton.jit
def grad_relu(x: tl.tensor):
    return tl.where(x > 0, 1, 0)

@triton.jit
def silu(x: tl.tensor):
    return x * tl.sigmoid(x)

@triton.jit
def grad_silu(x: tl.tensor):
    sig_x = tl.sigmoid(x)
    return sig_x + x * sig_x * (1 - sig_x)

@triton.jit
def approx_gelu(x: tl.tensor):
    # This is the approximation of gelu used by pytorch:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    pi: tl.constexpr = 3.14159265358979323846
    SQRT_2_DIV_PI: tl.constexpr = 0.7978845608
    tensor_cubed = x * x * x
    return 0.5 * x * (1 + libdevice.tanh(SQRT_2_DIV_PI * (x + 0.044715 * tensor_cubed)))

@triton.jit
def gelu(x: tl.tensor):
    # The actual GELU function:
    # https://github.com/pytorch/pytorch/blob/838f22c57df8d788a55a7637f93327f5ff26cd88/torch/_refs/nn/functional/__init__.py#L1058
    SQRT_1_DIV_2: tl.constexpr = 0.70710678118654752440
    return 0.5 * x * (1 + tl.erf(x * SQRT_1_DIV_2))

@triton.jit
def grad_gelu(x: tl.tensor):
    SQRT_1_DIV_2: tl.constexpr = 0.70710678118654752440
    SQRT_2_DIV_PI: tl.constexpr = 0.7978845608
    l = 1 + tl.erf(x * SQRT_1_DIV_2)
    z = x * SQRT_1_DIV_2
    r = x * SQRT_2_DIV_PI * tl.exp(-z * z)
    return 0.5 * (l + r)

@triton.jit
def swiglu(tile):
    tile = tl.reshape(tile, (tile.shape[0], tile.shape[1] // 2, 2))
    tile0, tile1 = tl.split(tile)
    tile0 = silu(tile0)
    return tile0 * tile1

@triton.jit
def grad_swiglu(tile):
    tile = tl.reshape(tile, (tile.shape[0], tile.shape[1] // 2, 2))
    tile0, tile1 = tl.split(tile)
    tile0 = grad_silu(tile0)
    return tile0 * tile1
    
@triton.jit
def geglu(tile):
    tile = tl.reshape(tile, (tile.shape[0], tile.shape[1] // 2, 2))
    tile0, tile1 = tl.split(tile)
    tile0 = gelu(tile0)
    return tile0 * tile1
    
TRITON_ACTIVATIONS = {
    "none": None,
    "relu": relu,
    "grad_relu": grad_relu,
    "silu": silu,
    "approx_gelu": approx_gelu,
    "gelu": gelu,
    "swiglu": swiglu,
    "geglu": geglu,
    "grad_silu": grad_silu,
    "grad_gelu": grad_gelu,
}