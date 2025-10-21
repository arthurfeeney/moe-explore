import triton
import triton.language as tl
from moe_explore.triton_kernels.activation import (
    relu,
    silu,
    gelu,
    swiglu,
    geglu,
    grad_relu,
    grad_silu,
    grad_gelu,
    grad_swiglu,
    grad_geglu
)

@triton.jit
def activation(x: tl.tensor, ACT: tl.constexpr):
    r"""
    torch.compile(fullgraph=True) on 2.8.0 complained about passing a triton
    function as a parameter to a kernel, so instead we pass in a string
    and determine the activation here.
    """
    if ACT == "relu":
        return relu(x)
    elif ACT == "silu":
        return silu(x)
    elif ACT == "gelu":
        return gelu(x)
    elif ACT == "swiglu":
        return swiglu(x)
    elif ACT == "geglu":
        return geglu(x)
    elif ACT == "grad_relu":
        return grad_relu(x)
    elif ACT == "grad_silu":
        return grad_silu(x)
    elif ACT == "grad_gelu":
        return grad_gelu(x)
    elif ACT == "grad_swiglu":
        return grad_swiglu(x)
    elif ACT == "grad_geglu":
        return grad_geglu(x)
    else:
        return x
    
@triton.jit
def epilogue_split(
    acc,
    EPILOGUE_SPLIT: tl.constexpr,
    EPILOGUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    r"""
    This breaks a 2d accumulator `acc` into `EPILOGUE_SPLIT` parts.
    and then applies the epilogue to each part. This assumes the
    epilogue function is splitable. I.e., element-wise operations
    or the GLUs should both work.
    if `EPILOGUE_SPLIT == 1`, then we just return a tuple with one element.
    """
    tl.static_assert(EPILOGUE_SPLIT == 1 or EPILOGUE_SPLIT == 2, "EPILOGUE_SPLIT must be 1 or 2")
    if EPILOGUE_SPLIT == 2:
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        accs = (activation(acc0, EPILOGUE), activation(acc1, EPILOGUE))
    else:
        accs = (activation(acc, EPILOGUE),)
    return accs

@triton.jit
def store_split_epilogue(
    out_ptrs,
    out_stride, # Stride of output along split axis.
    m_mask,
    N: tl.constexpr,
    accs,
):
    n_offset = tl.arange(0, accs[0].shape[1])
    for i in tl.static_range(len(accs)):
        out = accs[i]
        epilogue_split_offset = i * out.shape[1]
        tl.store(
            out_ptrs + epilogue_split_offset * out_stride, 
            out, 
            mask=m_mask[:, None] & (epilogue_split_offset + n_offset < N))