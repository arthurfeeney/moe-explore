from dataclasses import dataclass
from functools import partial
import torch
import triton
import triton.language as tl
from typing import Optional, Callable
from moe_explore.gpu_utils import get_gpu_sm_count
from .activation import TRITON_ACTIVATIONS
from .autotune_config import (
    AutotuneMode, 
    fast_autotune_configs, 
    max_autotune_configs
)

@dataclass
class MGroupedGEMMParams:
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
    activation: Optional[Callable] = None

@triton.jit
def m_grouped_gemm_inner(
    token_ptr,
    token_strides,
    weight_ptr,
    weight_strides,
    out_ptr,
    out_strides,
    permute_indices_ptr,
    problem_id,
    tile_id,
    last_problem_end,
    start_idx, 
    end_idx,
    m,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr,
    USE_TENSOR_DESCRIPTOR: tl.constexpr
):
    MASK_N: tl.constexpr = N % BLOCK_N != 0
    MASK_K: tl.constexpr = K % BLOCK_K != 0
    
    num_m_tiles = tl.cdiv(m, BLOCK_M)
    num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_N)
        
    num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
    while tile_id >= last_problem_end and tile_id < last_problem_end + num_tiles:
        tile_id_in_gemm = tile_id - last_problem_end
        tile_id_m = tile_id_in_gemm // num_n_tiles
        tile_id_n = tile_id_in_gemm % num_n_tiles
        
        # If GROUP_M is 0, don't do tile reordering.
        if GROUP_M == 0:
            tile_m_idx = (tile_id_in_gemm // num_n_tiles) * BLOCK_M
            tile_n_idx = (tile_id_in_gemm % num_n_tiles) * BLOCK_N
        else:
            # On 3.4.0, working around two potential compiler bugs. 
            # 1. triton doesn't like multiplying the group size with the num_n_tiles. Going through another
            # variable, group_m, gets it to compile. It also doesn't work to manually inline a group size.
            # 2. replacing with tl.swizzle2d hits a "failures [...] while processing an MLIR pass pipeline" 
            group_m = GROUP_M
            num_tiles_in_group = group_m * num_n_tiles
            group_id = tile_id_in_gemm // num_tiles_in_group
            first_id_m = group_id * group_m
            group_size_m = min(num_m_tiles - first_id_m, group_m)
            tile_m_idx = (first_id_m + ((tile_id_in_gemm % num_tiles_in_group) % group_size_m)) * BLOCK_M
            tile_n_idx = ((tile_id_in_gemm % num_tiles_in_group) // group_size_m) * BLOCK_N
        
        tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
        tile_n_offsets = tile_n_idx + tl.arange(0, BLOCK_N)
        tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets % m, BLOCK_M), BLOCK_M)
        tile_n_offsets = tl.max_contiguous(tl.multiple_of(tile_n_offsets % N, BLOCK_N), BLOCK_N)
        
        if GATHER_ROWS or SCATTER_ROWS:
            # Can avoid masking, since oversets are 0 <= tile_m_offsets < m
            permute_token_indices = tl.load(permute_indices_ptr + start_idx + tile_m_offsets)
            
        if GATHER_ROWS:
            token_indices = permute_token_indices // TOPK
        else:
            token_indices = start_idx + tile_m_offsets

        k_offset = tl.arange(0, BLOCK_K)

        token_row_offsets = token_indices * token_strides[0]
        token_col_offsets = k_offset * token_strides[1]
        token_ptrs = token_ptr + token_row_offsets[:, None] + token_col_offsets

        if USE_TENSOR_DESCRIPTOR:
            weight_desc = tl.make_tensor_descriptor(
                weight_ptr + problem_id * weight_strides[0],
                shape=(K, N),
                strides=(weight_strides[1], weight_strides[2]),
                block_shape=(BLOCK_K, BLOCK_N)
            )
        else:
            weight_problem_offset = problem_id * weight_strides[0]
            weight_row_offsets = k_offset[:, None] * weight_strides[1]
            weight_col_offsets = tile_n_offsets * weight_strides[2]
            weight_ptrs = weight_ptr + weight_problem_offset + weight_row_offsets + weight_col_offsets
        
        if MASK_N:
            n_mask = tile_n_offsets < N

        token_mask = start_idx + tile_m_offsets < end_idx
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER):
            tl.multiple_of(token_ptrs, [16, 16])
            tl.multiple_of(weight_ptrs, [16, 16])
            
            k_remaining = K - k * BLOCK_K
            if MASK_K and MASK_N:
                a_block = tl.load(token_ptrs, mask=token_mask[:, None] & (k_offset < k_remaining), other=0.0)
                b_block = tl.load(weight_ptrs, mask=n_mask[None, :] & (k_offset[:, None] < k_remaining), other=0.0)
            elif MASK_K:
                a_block = tl.load(token_ptrs, mask=token_mask[:, None] & (k_offset < k_remaining), other=0.0)
                b_block = tl.load(weight_ptrs, mask=k_offset[:, None] < k_remaining, other=0.0)
            elif MASK_N:
                a_block = tl.load(token_ptrs, mask=token_mask[:, None], other=0.0)
                b_block = tl.load(weight_ptrs, mask=n_mask[None, :], other=0.0)
            else:
                a_block = tl.load(token_ptrs, mask=token_mask[:, None], other=0.0)
                b_block = tl.load(weight_ptrs)
                
            acc = tl.dot(a_block, b_block, acc=acc)
            token_ptrs += BLOCK_K * token_strides[1]
            weight_ptrs += BLOCK_K * weight_strides[1]

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

        # The accumulators are all the same size, but the EPILOGUE may change the 
        # tile size in the N-dimension, so we use .shape[1], rather than BLOCK_N.
       # tile_n_idx = (tile_id_in_gemm % num_n_tiles) * accs[0].shape[1] * EPILOGUE_SPLIT
        out_tile_n_offsets = tile_n_idx // BLOCK_N * (accs[0].shape[1] * EPILOGUE_SPLIT) + tl.arange(0, accs[0].shape[1])
        
        if SCATTER_ROWS:         
            out_offsets = permute_token_indices[:, None] * out_strides[0] + out_tile_n_offsets * out_strides[1]
            out_ptrs = out_ptr + out_offsets
        else:
            out_row_offsets = start_idx + tile_m_offsets
            out_offsets = out_row_offsets[:, None] * out_strides[0] + out_tile_n_offsets * out_strides[1]
            out_ptrs = out_ptr + out_offsets

        for i in tl.static_range(len(accs)):
            out = accs[i]
            EPILOGUE_SPLIT_STEP = i * out.shape[1]
            if MASK_N:
                tl.store(
                    out_ptrs + EPILOGUE_SPLIT_STEP * out_strides[1], 
                    out, 
                    mask=token_mask[:, None] & (out_tile_n_offsets + EPILOGUE_SPLIT_STEP < N), 
                    cache_modifier=".cs")
            else:
                tl.store(
                    out_ptrs + EPILOGUE_SPLIT_STEP * out_strides[1], 
                    out, 
                    mask=token_mask[:, None], 
                    cache_modifier=".cs")

        tile_id += NUM_PROGRAMS
    return tile_id, num_tiles

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
    NUM_TOKENS: tl.constexpr,
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
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr,
    USE_TENSOR_DESCRIPTOR: tl.constexpr
):
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

    start_idx = 0
    for problem_id in tl.range(0, E):
        end_idx = tl.load(group_indices_ptr + problem_id + 1, cache_modifier=".ca")
        m = end_idx - start_idx
        
        tl.assume(start_idx >= 0)
        tl.assume(end_idx >= start_idx)
        tl.assume(m >= 0)
        tl.assume(m <= NUM_TOKENS * TOPK)
        
        tile_id, num_tiles = m_grouped_gemm_inner(
            token_ptr,
            token_strides,
            weight_ptr,
            weight_strides,
            out_ptr,
            out_strides,
            permute_indices_ptr,
            problem_id,
            tile_id,
            last_problem_end,
            start_idx,
            end_idx,
            m,
            N,
            K,
            TOPK,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            ACC_DTYPE,
            GATHER_ROWS,
            SCATTER_ROWS,
            NUM_PROGRAMS,
            GROUP_M,
            EPILOGUE,
            EPILOGUE_SPLIT,
            DISALLOW_ACC_MULTI_BUFFER,
            USE_TENSOR_DESCRIPTOR
        )
        
        start_idx = end_idx
        last_problem_end += num_tiles 

_fast_autotune_m_grouped_gemm_persistent_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=['NUM_TOKENS', 'E', 'N', 'K', 'GATHER_ROWS', 'SCATTER_ROWS'],
    reset_to_zero=['out_ptr']
)(m_grouped_gemm_persistent_kernel)

_max_autotune_m_grouped_gemm_persistent_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True),
    key=['NUM_TOKENS', 'E', 'N', 'K', 'GATHER_ROWS', 'SCATTER_ROWS'],
    reset_to_zero=['out_ptr']
)(m_grouped_gemm_persistent_kernel)

def m_grouped_gemm_default_config(params):
    e = params.weight.size(0)
    avg_tokens_per_expert = triton.cdiv(params.num_tokens * params.topk, e)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    if not (params.gather or params.scatter):
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "GROUP_M": 6,
                "EPILOGUE_SPLIT": 2,
                "DISALLOW_ACC_MULTI_BUFFER": False,
                "USE_TENSOR_DESCRIPTOR": False,
            },
            num_warps=8, 
            num_stages=4
        )
    elif params.gather:
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": 64, 
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "GROUP_M": 6,
                "EPILOGUE_SPLIT": 2,
                "DISALLOW_ACC_MULTI_BUFFER": False,
                "USE_TENSOR_DESCRIPTOR": False,
            },
            num_warps=8, 
            num_stages=4
        )
    elif params.scatter:
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "GROUP_M": 6,
                "EPILOGUE_SPLIT": 2,
                "DISALLOW_ACC_MULTI_BUFFER": False,
                "USE_TENSOR_DESCRIPTOR": False,
            },
            num_warps=8, 
            num_stages=4
        )
    return default_config

def m_grouped_gemm(
    token: torch.Tensor,
    params: MGroupedGEMMParams,
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

    if params.gather or params.scatter:
        out_rows = num_tokens * params.topk
    else:
        out_rows = num_tokens
        
    out_cols = n
    if params.activation is not None and "glu" in params.activation:
        out_cols //= 2
        
    out = torch.empty((out_rows, out_cols), device=token.device, dtype=token.dtype)

    default_config = m_grouped_gemm_default_config(params)
    default_kwargs = default_config.all_kwargs()
    func = m_grouped_gemm_persistent_kernel
    if autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_m_grouped_gemm_persistent_kernel
        default_kwargs = {}
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_m_grouped_gemm_persistent_kernel
        default_kwargs = {}
        
    epilogue = TRITON_ACTIVATIONS[params.activation] if params.activation in TRITON_ACTIVATIONS else None
        
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
        NUM_TOKENS=num_tokens,
        ACC_DTYPE=tl.float32,
        E=e, 
        K=k,
        N=n,
        TOPK=params.topk,
        GATHER_ROWS=params.gather,
        SCATTER_ROWS=params.scatter,
        EPILOGUE=epilogue,
        **default_kwargs
    )

    return out