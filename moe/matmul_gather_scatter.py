import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def matmul_gather_scatter_kernel(
    # a is a [t, K] tensor.
    a_ptr,
    # b is [E, K, N]
    b_ptr,
    # c is a [G, N] tensor (G = g1 + g2 + ...)
    c_ptr,
    # The group indices essentially determime M sizes.
    # [0, g1, g2, ..., gn]. So M_n = gn - g{n-1}
    group_indices_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    scale_ptr,
    # b is [E, K, N],
    # TODO: b could also be transposed...
    # E determines the number of problems.
    E: tl.constexpr,
    # size of A before scatter
    K: tl.constexpr,
    N: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    last_problem_end = 0

    for problem_id in range(0, E):
        end_idx = tl.load(group_indices_ptr + problem_id + 1)
        start_idx = tl.load(group_indices_ptr + problem_id)
        m = end_idx - start_idx

        # TODO: These should be set based on `m`
        BLOCK_M: tl.constexpr = 16
        BLOCK_N: tl.constexpr = 16
        BLOCK_K: tl.constexpr = 16

        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
        while (tile_id >= last_problem_end and tile_id < last_problem_end + num_tiles):
            tile_id_in_gemm = tile_id - last_problem_end
            tile_m_idx = (tile_id_in_gemm // num_n_tiles) * BLOCK_M
            tile_n_idx = (tile_id_in_gemm % num_n_tiles) * BLOCK_N

            tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
            tile_n_offsets = tile_n_idx + tl.arange(0, BLOCK_N)

            # mask based on group size, used to mask load of group indices and store to `c`
            gather_indices_offsets = start_idx + tile_m_offsets
            gather_mask = gather_indices_offsets < end_idx

            if GATHER_ROWS:
                a_row_indices = tl.load(gather_indices_ptr + gather_indices_offsets, mask=gather_mask)
            else:
                a_row_indices = tile_m_offsets

            a_ptrs = a_ptr + (a_row_indices * K)[:, None] + tl.arange(0, BLOCK_K)

            b_problem_offset = problem_id * K * N
            b_row_offsets = tl.arange(0, BLOCK_K)[:, None] * N
            b_col_offsets = tile_n_offsets 
            b_ptrs = b_ptr + b_problem_offset + b_row_offsets + b_col_offsets 

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                # we gather some garbage rows, so have to mask part of a
                a_block = tl.load(a_ptrs, mask=gather_mask[:, None])
                b_block = tl.load(b_ptrs)
                acc = tl.dot(a_block, b_block, acc=acc) 
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * N

            acc = tl.cast(acc, tl.float16)

            if SCATTER_ROWS:
                pass
            else:
                if GATHER_ROWS:
                    c_mask = gather_mask[:, None]
                else:
                    c_mask = None
                c_row_offsets = start_idx + tile_m_offsets #tile_m_idx + tl.arange(0, BLOCK_M)[:, None]
                c_col_offsets = tile_n_offsets #tile_n_idx + tl.arange(0, BLOCK_N)
                c_offsets = c_row_offsets[:, None] * N + c_col_offsets
                c_ptrs = c_ptr + c_offsets
                tl.store(c_ptrs, acc, mask=gather_mask[:, None])

            tile_id += NUM_PROGRAMS
        last_problem_end += num_tiles 

def matmul_gather_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    group_indices: torch.Tensor,
    gather_indices: Optional[torch.Tensor],
    scatter_indices: Optional[torch.Tensor],
    scale: Optional[torch.Tensor]
):
    assert a.dim() == 2
    assert b.dim() == 3
    assert a.size(1) == b.size(1)
    assert group_indices.dim() == 1
    assert gather_indices is None or gather_indices.dim() == 1
    assert scatter_indices is None or scatter_indices.dim() == 1

    e, k, n = b.size()

    if scatter_indices is None and gather_indices is not None:
        c_rows = gather_indices.size(0)
    else:
        c_rows = a.size(0) 

    c = torch.zeros((c_rows, n), device=a.device, dtype=a.dtype)
    num_programs = 1
    matmul_gather_scatter_kernel[(num_programs,)](
        a, 
        b, 
        c, 
        group_indices, 
        gather_indices, 
        scatter_indices,
        scale,
        E=e, 
        K=k, 
        N=n,
        GATHER_ROWS=gather_indices is not None,
        SCATTER_ROWS=scatter_indices is not None,
        NUM_PROGRAMS=num_programs
    )

    return c
