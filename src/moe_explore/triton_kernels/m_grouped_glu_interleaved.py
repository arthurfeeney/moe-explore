from dataclasses import dataclass
from functools import partial
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
from .activation import TRITON_ACTIVATIONS

@dataclass
class MGroupedGLUInterleavedParams:
    weight: torch.Tensor
    group_indices: torch.Tensor
    permute_indices: Optional[torch.Tensor]
    gather: bool
    num_tokens: int
    topk: int
    activation: str

@triton.jit
def m_grouped_glu_interleaved_persistent_kernel(
    token_ptr,
    token_strides,
    weight_ptr,
    weight_strides,
    out_ptr,
    out_strides,
    group_indices_ptr,
    permute_indices_ptr,
    ACTIVATION: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    tl.static_assert(K % BLOCK_K == 0, "K must be a multiple of BLOCK_K")
    tl.static_assert(N % BLOCK_N == 0, "N must be a multiple of BLOCK_N")
    tl.static_assert((N // 2) % BLOCK_N == 0, "N must be a multiple of 2")
    
    # Compiler error with triton 3.3.1 when defining constexpr inside a while loop
    OUTPUT_BLOCK_N: tl.constexpr = BLOCK_N // 2

    tile_id = tl.program_id(axis=0)
    last_problem_end = 0

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
         
            out_tile_n_idx = (tile_id_in_gemm % num_n_tiles) * OUTPUT_BLOCK_N
            out_tile_n_offsets = out_tile_n_idx + tl.arange(0, OUTPUT_BLOCK_N)
         
            token_mask = start_idx + tile_m_offsets < end_idx

            if GATHER_ROWS:
                permute_token_indices = tl.load(permute_indices_ptr + start_idx + tile_m_offsets,
                                                mask=token_mask, 
                                                other=0)
                token_indices = permute_token_indices // TOPK
            else:
                token_indices = start_idx + tile_m_offsets

            k_offset = tl.arange(0, BLOCK_K)

            token_row_offsets = token_indices * token_strides[0]
            token_col_offsets = k_offset * token_strides[1]
            token_ptrs = token_ptr + token_row_offsets[:, None] + token_col_offsets

            weight_problem_offset = problem_id * weight_strides[0]
            weight_row_offsets = k_offset[:, None] * weight_strides[1]
            weight_col_offsets = tile_n_offsets * weight_strides[2]

            weight_ptrs = weight_ptr + weight_problem_offset + weight_row_offsets + weight_col_offsets

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                tl.multiple_of(token_ptrs, [16, 16])
                tl.multiple_of(weight_ptrs, [16, 16])
                
                tokens_block = tl.load(token_ptrs, mask=token_mask[:, None], other=0.0)                
                weights_block = tl.load(weight_ptrs)
                
                acc = tl.dot(tokens_block, weights_block, acc=acc)
                
                token_ptrs += BLOCK_K * token_strides[1]
                weight_ptrs += BLOCK_K * weight_strides[1]

            out_tile_n_idx = (tile_id_in_gemm % num_n_tiles) * (BLOCK_N // 2)
            tile_n_offsets0 = out_tile_n_idx + tl.arange(0, BLOCK_N // 4)
            tile_n_offsets1 = out_tile_n_idx + BLOCK_N // 4

            out_row_offsets = start_idx + tile_m_offsets
            out_offsets = out_row_offsets[:, None] * out_strides[0] + tile_n_offsets0 * out_strides[1]
            out_ptrs = out_ptr + out_offsets

            # Note, there are multiple splits. One corresponds to epilogue splitting,
            # and the others are to separate the interleaved the outputs for the
            # `gate_weight` and `up_weight`.
                
            # Epilogue splitting: Compute epilogue on two halves of the tile.
            acc = acc.reshape(BLOCK_M, 2, BLOCK_N // 2)
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            
            # Split again on the interleaved weights.
            acc = acc0.reshape(BLOCK_M, BLOCK_N // 4, 2)
            acc_gate, acc_up = acc.split()
            acc_gate = ACTIVATION(acc_gate)
            acc_gate = acc_gate.to(out_ptr.dtype.element_ty)
            out_tile0 = acc_gate * acc_up
            tl.store(out_ptrs, out_tile0, mask=token_mask[:, None])
            
            acc = acc1.reshape(BLOCK_M, BLOCK_N // 4, 2)
            acc_gate, acc_up = acc.split()
            acc_gate = ACTIVATION(acc_gate)
            acc_gate = acc_gate.to(out_ptr.dtype.element_ty)
            out_tile1 = acc_gate * acc_up
            tl.store(out_ptrs + (BLOCK_N // 4) * out_strides[1], out_tile1, mask=token_mask[:, None])

            tile_id += NUM_PROGRAMS
        last_problem_end += num_tiles 

_fast_autotune_m_grouped_glu_interleaved_persistent_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=["E", "N", "K", "GATHER_ROWS"],
    reset_to_zero=['out_ptr']
)(m_grouped_glu_interleaved_persistent_kernel)

_max_autotune_m_grouped_glu_interleaved_persistent_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True),
    key=["E", "N", "K", "GATHER_ROWS"],
    reset_to_zero=['out_ptr']
)(m_grouped_glu_interleaved_persistent_kernel)

def m_grouped_glu_interleaved_default_config(params):
    e = params.weight.size(0)
    avg_tokens_per_expert = triton.cdiv(params.num_tokens * params.topk, e)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    
    if not params.gather:
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count()
            },
            num_warps=8, 
            num_stages=5
        )
    else:
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count()
            },
            num_warps=8, 
            num_stages=5
        )
    return default_config

def m_grouped_glu_interleaved(
    token: torch.Tensor,
    params: MGroupedGLUInterleavedParams,
    autotune_mode: Optional[AutotuneMode] = None
):
    r"""
    This kernel assumes that the GLU weights in `params.weight` are inteleaved.
    So that `params.weight[i] is gate_weight[i // 2] and if i is even, and up_weight[i] otherwise.
    """
    assert token.dim() == 2
    assert params.weight.dim() == 3
    assert autotune_mode is None or autotune_mode in AutotuneMode
    assert params.activation in ("silu", "gelu")
    if params.gather:
        assert params.permute_indices is not None

    num_tokens = params.num_tokens
    e, k, n = params.weight.size()
    
    if params.gather:
        out_rows = num_tokens * params.topk
    else:
        out_rows = num_tokens

    out = torch.empty((out_rows, n // 2), device=token.device, dtype=token.dtype)
    
    default_config = m_grouped_glu_interleaved_default_config(params)
    default_kwargs = default_config.all_kwargs()
    func = m_grouped_glu_interleaved_persistent_kernel
    if autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_m_grouped_glu_interleaved_persistent_kernel
        default_kwargs = {}
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_m_grouped_glu_interleaved_persistent_kernel
        default_kwargs = {}
    
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
        ACTIVATION=TRITON_ACTIVATIONS[params.activation],
        ACC_DTYPE=tl.float32,
        E=e, 
        K=k,
        N=n,
        TOPK=params.topk,
        GATHER_ROWS=params.gather,
        **default_kwargs
    )

    return out