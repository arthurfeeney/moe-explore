import triton
import triton.language as tl

@triton.jit
def epilogue_split(
    acc,
    EPILOGUE_SPLIT: tl.constexpr,
    EPILOGUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    tl.static_assert(EPILOGUE_SPLIT == 1 or EPILOGUE_SPLIT == 2, "EPILOGUE_SPLIT must be 1 or 2")
    if EPILOGUE_SPLIT == 2:
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        accs = (acc0, acc1)
    else:
        accs = (acc,)
    
    for i in tl.static_range(len(accs)):
        if EPILOGUE is not None:
            accs[i] = EPILOGUE(accs[i])

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
            mask=m_mask[:, None] & (epilogue_split_offset + n_offset < N),
            cache_modifier=".cs")