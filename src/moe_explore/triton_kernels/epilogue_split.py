import triton
import triton.language as tl
from moe_explore.triton_kernels.activation import activation
    
@triton.jit
def epilogue_split(
    acc,
    EPILOGUE_SPLIT: tl.constexpr,
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
        accs = (acc0, acc1)
    else:
        accs = (acc,)
    return accs

@triton.jit
def store_split_epilogue(
    ptrs,
    split_stride, # Stride of output along split axis.
    m_mask,
    n_remaining,
    accs,
    EPILOGUE: tl.constexpr,
    preact_for_grad_ptrs,
):
    for i in tl.static_range(len(accs)):
        out = activation(accs[i], EPILOGUE) if EPILOGUE is not None else accs[i]
        n_offset = tl.arange(0, out.shape[1])
        epilogue_split_offset = i * out.shape[1]
        tl.store(
            ptrs + epilogue_split_offset * split_stride, 
            out, 
            mask=m_mask[:, None] & (epilogue_split_offset + n_offset < n_remaining))