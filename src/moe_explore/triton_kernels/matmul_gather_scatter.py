import itertools
import torch
import triton
import triton.language as tl
from typing import Optional
from moe_explore.gpu_utils import get_gpu_sm_count
from moe_explore.autotune_config import AutotuneMode, fast_autotune_configs, max_autotune_configs

@triton.jit
def matmul_gather_scatter_kernel(
    # a is a [t, K] tensor, always row-major since
    # we want to gather/scatter rows.
    a_ptr,
    # b is [E, K, N]
    # TODO: Should also setup so b can be transposed
    b_ptr,
    c_ptr,
    # The group indices essentially determime M size of each problem.
    # [0, g1, g2, ..., gn]. So Mn = gn - g{n-1}
    group_indices_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    # Scales use a different set of indices
    scales_ptr,
    scales_indices_ptr,
    # E determines the number of problems.
    E: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    SCALE_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    tl.static_assert(K % BLOCK_K == 0)
    tl.static_assert(N % BLOCK_N == 0)

    tile_id = tl.program_id(axis=0)
    last_problem_end = 0

    for problem_id in range(0, E):
        end_idx = tl.load(group_indices_ptr + problem_id + 1)
        start_idx = tl.load(group_indices_ptr + problem_id)
        m = end_idx - start_idx

        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
        while (tile_id >= last_problem_end and tile_id < last_problem_end + num_tiles):
            tile_id_in_gemm = tile_id - last_problem_end
            tile_m_idx = (tile_id_in_gemm // num_n_tiles) * BLOCK_M
            tile_n_idx = (tile_id_in_gemm % num_n_tiles) * BLOCK_N
            tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
            tile_n_offsets = tile_n_idx + tl.arange(0, BLOCK_N)

            gather_scatter_indices_offsets = start_idx + tile_m_offsets
            # mask is based on group size, not number of rows of a 
            gather_scatter_mask = gather_scatter_indices_offsets < end_idx

            if GATHER_ROWS:
                a_row_indices = tl.load(gather_indices_ptr + gather_scatter_indices_offsets,
                                        mask=gather_scatter_mask, 
                                        other=0)
            else:
                a_row_indices = start_idx + tile_m_offsets

            a_ptrs = a_ptr + (a_row_indices * K)[:, None] + tl.arange(0, BLOCK_K)

            b_problem_offset = problem_id * K * N
            b_row_offsets = tl.arange(0, BLOCK_K)[:, None] * N
            b_col_offsets = tile_n_offsets 
            b_ptrs = b_ptr + b_problem_offset + b_row_offsets + b_col_offsets 

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                # Potentially gather some garbage rows, so zero-mask some rows of a
                a_block = tl.load(a_ptrs, mask=gather_scatter_mask[:, None], other=0.0)
                b_block = tl.load(b_ptrs)
                acc = tl.dot(a_block, b_block, acc=acc) 
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * N
            acc = tl.cast(acc, tl.float16)

            c_col_offsets = tile_n_offsets
            if SCATTER_ROWS:
                if SCALE_ROWS:
                    scales_indices = tl.load(scales_indices_ptr + gather_scatter_indices_offsets, 
                                             mask=gather_scatter_mask,
                                             other=0)
                    scales = tl.load(scales_ptr + scales_indices,
                                     mask=gather_scatter_mask, 
                                     other=0.0)
                    acc = acc * scales[:, None]
                c_row_offsets = tl.load(scatter_indices_ptr + gather_scatter_indices_offsets, 
                                        mask=gather_scatter_mask, 
                                        other=0)
                c_offsets = c_row_offsets[:, None] * N + c_col_offsets
                c_ptrs = c_ptr + c_offsets
                # TODO: 
                # 1. c needs to be initialized for this to work. Tricky to just zero 
                #    initialize per-tile, because we write to scattered spots
                # 2. How is scatter-reduce done in pytorch?
                # 3. There are a lot of stalls here. Mostly scoreboard, some from memory barrier
                tl.atomic_add(c_ptrs, acc, mask=gather_scatter_mask[:, None])
            else:
                # if we aren't scattering, just write output
                c_row_offsets = start_idx + tile_m_offsets
                c_offsets = c_row_offsets[:, None] * N + c_col_offsets
                c_ptrs = c_ptr + c_offsets
                tl.store(c_ptrs, acc, mask=gather_scatter_mask[:, None])

            tile_id += NUM_PROGRAMS
        last_problem_end += num_tiles 

_fast_autotune_matmul_gather_scatter_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=['E', 'N', 'K'],
    reset_to_zero=['c_ptr']
)(matmul_gather_scatter_kernel)

_max_autotune_matmul_gather_scatter_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True),
    key=['E', 'N', 'K'],
    reset_to_zero=['c_ptr']
)(matmul_gather_scatter_kernel)

def get_output_rows(a_rows, gather_indices, scatter_indices, output_rows):
    if output_rows is not None:
        return output_rows
    if gather_indices is not None and scatter_indices is None:
        return gather_indices.size(0)
    if gather_indices is None and scatter_indices is not None:
        # this is a kernel call, which can be avoided by passing in output_rows
        return scatter_indices.max() + 1
    # if we don't scatter or gather, then it's just a normal matmul,
    # so size of output matches input
    return a_rows

def matmul_gather_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    group_indices: torch.Tensor,
    gather_indices: Optional[torch.Tensor] = None,
    scatter_indices: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    scales_indices: Optional[torch.Tensor] = None,
    output_rows: Optional[int] = None,
    autotune_mode: Optional[AutotuneMode] = None
):
    assert a.dim() == 2
    assert b.dim() == 3
    assert a.size(1) == b.size(1)
    assert group_indices is not None
    assert group_indices.dim() == 1
    assert gather_indices is None or gather_indices.dim() == 1
    assert scatter_indices is None or scatter_indices.dim() == 1
    assert autotune_mode is None or autotune_mode in AutotuneMode

    e, k, n = b.size()

    default_kwargs = {}
    if autotune_mode is None or autotune_mode == AutotuneMode.NONE:
        default_kwargs = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_PROGRAMS": get_gpu_sm_count()}
        func = matmul_gather_scatter_kernel
    elif autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_matmul_gather_scatter_kernel
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_matmul_gather_scatter_kernel

    c_rows = get_output_rows(a.size(0), gather_indices, scatter_indices, output_rows)
    c = torch.zeros((c_rows, n), device=a.device, dtype=a.dtype)
    grid = lambda META: (META["NUM_PROGRAMS"],)
    func[grid](
        a, 
        b, 
        c, 
        group_indices, 
        gather_indices, 
        scatter_indices,
        scales,
        scales_indices,
        E=e, 
        K=k, 
        N=n,
        GATHER_ROWS=gather_indices is not None,
        SCATTER_ROWS=scatter_indices is not None,
        SCALE_ROWS=scales is not None,
        **default_kwargs
    )

    return c
