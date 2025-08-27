import torch
import triton
import triton.language as tl

@triton.heuristics(values={
    'BLOCK_SIZE_DIM1': lambda args: 64,
    'BLOCK_SIZE_DIM2': lambda args: 64
})
@triton.jit
def row_gather_kernel(
    tokens_ptr,
    tokens_strides,
    gather_indices_ptr,
    output_ptr,
    output_strides,
    OUTPUT_NUM_ROWS: tl.constexpr,
    OUTPUT_NUM_COLS: tl.constexpr,
    BLOCK_SIZE_DIM1: tl.constexpr,
    BLOCK_SIZE_DIM2: tl.constexpr,
):
    pid_dim1 = tl.program_id(0)
    pid_dim2 = tl.program_id(1)
    
    output_row_offsets = (pid_dim1 * BLOCK_SIZE_DIM1) + tl.arange(0, BLOCK_SIZE_DIM1)
    gather_row_indices = tl.load(gather_indices_ptr + output_row_offsets, mask=output_row_offsets < OUTPUT_NUM_ROWS, other=0)
    col_offsets = pid_dim2 * BLOCK_SIZE_DIM2 + tl.arange(0, BLOCK_SIZE_DIM2)

    tokens_offsets = (gather_row_indices * tokens_strides[0])[:, None] + col_offsets * tokens_strides[1]
    tokens_ptrs = tokens_ptr + tokens_offsets
    tokens = tl.load(tokens_ptrs, mask=(col_offsets < OUTPUT_NUM_COLS)[None, :], other=0.0)

    output_offsets = (output_row_offsets * output_strides[0])[:, None] + col_offsets * output_strides[1]
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, tokens, mask=(output_row_offsets < OUTPUT_NUM_ROWS)[:, None] and (col_offsets < OUTPUT_NUM_COLS))

def row_gather(
    tokens: torch.Tensor,
    gather_indices: torch.Tensor,
):
    assert tokens.ndim == 2
    assert gather_indices.ndim == 1
    
    output = torch.empty((gather_indices.size(0), tokens.size(1)), device=tokens.device, dtype=tokens.dtype)
    grid = lambda META: (
        triton.cdiv(output.size(0), META["BLOCK_SIZE_DIM1"]),
        triton.cdiv(output.size(1), META["BLOCK_SIZE_DIM2"])
    )
    row_gather_kernel[grid](
        tokens,
        tokens.stride(),
        gather_indices,
        output,
        output.stride(),
        output.size(0),
        output.size(1)
    )
    
    return output

@triton.heuristics(values={
    'BLOCK_SIZE_DIM1': lambda args: 64,
    'BLOCK_SIZE_DIM2': lambda args: 64
})
@triton.jit
def row_scatter_kernel(
    tokens_ptr,
    tokens_strides,
    scatter_indices_ptr,
    output_ptr,
    output_strides,
    OUTPUT_NUM_ROWS: tl.constexpr,
    OUTPUT_NUM_COLS: tl.constexpr,
    BLOCK_SIZE_DIM1: tl.constexpr,
    BLOCK_SIZE_DIM2: tl.constexpr,
):
    pid_dim1 = tl.program_id(0)
    pid_dim2 = tl.program_id(1)
    
    row_offsets = pid_dim1 * BLOCK_SIZE_DIM1 + tl.arange(0, BLOCK_SIZE_DIM1)
    col_offsets = pid_dim2 * BLOCK_SIZE_DIM2 + tl.arange(0, BLOCK_SIZE_DIM2)

    tokens_offsets = (row_offsets * tokens_strides[0])[:, None] + col_offsets * tokens_strides[1]
    tokens_ptrs = tokens_ptr + tokens_offsets
    tokens = tl.load(tokens_ptrs, mask=(row_offsets < OUTPUT_NUM_ROWS)[:, None] and (col_offsets < OUTPUT_NUM_COLS), other=0.0)

    scatter_row_indices = tl.load(scatter_indices_ptr + row_offsets, mask=row_offsets < OUTPUT_NUM_ROWS, other=0)
    output_offsets = (scatter_row_indices * output_strides[0])[:, None] + col_offsets * output_strides[1]
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, tokens, mask=(row_offsets < OUTPUT_NUM_ROWS)[:, None] and (col_offsets < OUTPUT_NUM_COLS))

def row_scatter(
    tokens: torch.Tensor,
    scatter_indices: torch.Tensor,
):
    r"""
    Note: this assumes that the scatter_indices are all unique.
    """
    assert tokens.ndim == 2
    assert scatter_indices.ndim == 1
    
    output = torch.empty_like(tokens)
    grid = lambda META: (
        triton.cdiv(output.size(0), META["BLOCK_SIZE_DIM1"]),
        triton.cdiv(output.size(1), META["BLOCK_SIZE_DIM2"])
    )
    row_scatter_kernel[grid](
        tokens,
        tokens.stride(),
        scatter_indices,
        output,
        output.stride(),
        output.size(0),
        output.size(1)
    )
    
    return output