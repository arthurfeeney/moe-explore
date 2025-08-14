from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from typing import Optional
from moe_explore.gpu_utils import get_gpu_sm_count
from .autotune_config import (
    AutotuneMode, 
    fast_autotune_configs, 
    max_autotune_configs
)

@dataclass
class FusedMoeParams:
    weight: torch.Tensor
    group_indices: torch.Tensor
    permute_indices: Optional[torch.Tensor]
    gather: bool
    scatter: bool
    # `num_tokens` is confusing. When doing a scatter or gather, it's the number of tokens
    # before routing! If we are NOT doing a scatter or gather, it's just the number of tokens being input.
    # If we do a gather, `num_tokens` is the number of tokens in the input, before the gather.
    # Similar for when we do a scatter, it's the number of tokens in the output, after the scatter.
    num_tokens: int
    topk: int
    scales: Optional[torch.Tensor]

@triton.jit
def m_grouped_gemm_persistent_kernel(
    token_ptr,
    token_strides,
    weight_ptr,
    weight_strides,
    out_ptr,
    out_strides,
    group_indices_ptr,
    permute_indices_ptr,
    ACC_DTYPE: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    MASK_K: tl.constexpr = K % BLOCK_K != 0
    MASK_N: tl.constexpr = N % BLOCK_N != 0

    tile_id = tl.program_id(axis=0)
    last_problem_end = 0
    
    tl.assume(tile_id >= 0)
    tl.assume(token_strides[0] > 0)
    tl.assume(token_strides[1] > 0)
    tl.assume(weight_strides[0] > 0)
    tl.assume(weight_strides[1] > 0)
    tl.assume(weight_strides[2] > 0)
    tl.assume(out_strides[0] > 0)
    tl.assume(out_strides[1] > 0)

    for problem_id in tl.range(0, E, flatten=True):
        end_idx = tl.load(group_indices_ptr + problem_id + 1)
        start_idx = tl.load(group_indices_ptr + problem_id)
        m = end_idx - start_idx
        
        tl.assume(start_idx >= 0)
        tl.assume(end_idx >= start_idx)
        tl.assume(m >= 0)
        
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
        while (tile_id >= last_problem_end and tile_id < last_problem_end + num_tiles):
            tile_id_in_gemm = tile_id - last_problem_end
            tile_m_idx = (tile_id_in_gemm // num_n_tiles) * BLOCK_M
            tile_n_idx = (tile_id_in_gemm % num_n_tiles) * BLOCK_N
            tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
            tile_n_offsets = tile_n_idx + tl.arange(0, BLOCK_N)

            if GATHER_ROWS or SCATTER_ROWS:
                permute_indices_offsets = start_idx + tile_m_offsets
                token_mask = permute_indices_offsets < end_idx
                permute_token_indices = tl.load(permute_indices_ptr + permute_indices_offsets,
                                                mask=token_mask, 
                                                other=0)
            
            if GATHER_ROWS:
                token_indices = permute_token_indices // TOPK
            else:
                token_mask = start_idx + tile_m_offsets < end_idx
                token_indices = start_idx + tile_m_offsets

            token_row_offsets = token_indices * token_strides[0]
            token_col_offsets = tl.arange(0, BLOCK_K) * token_strides[1]
            token_ptrs = token_ptr + token_row_offsets[:, None] + token_col_offsets

            weight_problem_offset = problem_id * weight_strides[0]
            weight_row_offsets = tl.arange(0, BLOCK_K)[:, None] * weight_strides[1]
            weight_col_offsets = tile_n_offsets * weight_strides[2]
            weight_ptrs = weight_ptr + weight_problem_offset + weight_row_offsets + weight_col_offsets
            
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                tl.multiple_of(token_ptrs, [16, 16])
                tl.multiple_of(weight_ptrs, [16, 16])
                a_block = tl.load(token_ptrs, mask=token_mask[:, None], other=0.0)
                b_block = tl.load(weight_ptrs)
                acc = tl.dot(a_block, b_block, acc=acc) 
                token_ptrs += BLOCK_K * token_strides[1]
                weight_ptrs += BLOCK_K * weight_strides[1]

            acc = acc.to(out_ptr.dtype.element_ty)

            if SCATTER_ROWS:
                out_offsets = permute_token_indices[:, None] * out_strides[0] + tile_n_offsets * out_strides[1]
                out_ptrs = out_ptr + out_offsets
                tl.store(out_ptrs, acc, mask=token_mask[:, None])
            else:
                out_row_offsets = start_idx + tile_m_offsets
                out_offsets = out_row_offsets[:, None] * out_strides[0] + tile_n_offsets * out_strides[1]
                out_ptrs = out_ptr + out_offsets
                tl.store(out_ptrs, acc, mask=token_mask[:, None])

            tile_id += NUM_PROGRAMS
        last_problem_end += num_tiles 

_fast_autotune_m_grouped_gemm_persistent_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=['E', 'N', 'K'],
    reset_to_zero=['out_ptr']
)(m_grouped_gemm_persistent_kernel)

_max_autotune_m_grouped_gemm_persistent_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True),
    key=['E', 'N', 'K'],
    reset_to_zero=['out_ptr']
)(m_grouped_gemm_persistent_kernel)

@torch.compile
def scale_and_reduce(out, scales, num_tokens, topk, n):
    return (out.view(num_tokens, topk, n) * scales[..., None]).sum(1)

def fused_moe(
    token: torch.Tensor,
    params: FusedMoeParams,
    autotune_mode: Optional[AutotuneMode] = None
):
    assert token.dim() == 2
    assert params.weight.dim() == 3
    assert token.size(1) == params.weight.size(1)
    assert autotune_mode is None or autotune_mode in AutotuneMode

    if params.gather or params.scatter:
        assert params.permute_indices is not None

    num_tokens = params.num_tokens
    e, k, n = params.weight.size()

    default_kwargs = {}
    if autotune_mode is None or autotune_mode == AutotuneMode.NONE:
        if not params.scatter:
            default_kwargs = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "NUM_PROGRAMS": get_gpu_sm_count(), "num_warps": 8, "num_stages": 4}
        elif params.scatter:
            default_kwargs = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_PROGRAMS": 4 * get_gpu_sm_count(), "num_warps": 4, "num_stages": 4}
        func = m_grouped_gemm_persistent_kernel
    elif autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_m_grouped_gemm_persistent_kernel
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_m_grouped_gemm_persistent_kernel

    if params.gather or params.scatter:
        c_rows = num_tokens * params.topk
    else:
        c_rows = num_tokens

    out = torch.empty((c_rows, n), device=token.device, dtype=token.dtype)
    
    grid = lambda META: (META["NUM_PROGRAMS"],)
        
    func[grid](
        token, 
        token.stride(),
        params.weight,
        params.weight.stride(),
        out,
        out.stride(),
        params.group_indices, 
        params.permute_indices, 
        ACC_DTYPE=tl.float32,
        E=e, 
        K=k,
        N=n,
        TOPK=params.topk,
        GATHER_ROWS=params.gather,
        SCATTER_ROWS=params.scatter,
        **default_kwargs
    )

    if params.scales is not None and params.scatter:
        out = scale_and_reduce(out, params.scales, num_tokens, params.topk, n)

    return out