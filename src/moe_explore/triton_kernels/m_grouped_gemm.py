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
from .epilogue_split import epilogue_split, store_split_epilogue

@dataclass
class MGroupedGEMMParams:
    permute_indices: Optional[torch.Tensor]
    gather: bool
    scatter: bool
    # `num_tokens` is confusing. When doing a scatter or gather, it's the number of tokens
    # before routing! If we are NOT doing a scatter or gather, it's just the number of tokens being input.
    # If we do a gather, `num_tokens` is the number of tokens in the input, before the gather.
    # Similar for when we do a scatter, it's the number of tokens in the output, after the scatter.
    num_tokens: int
    topk: int
    # This is necessary for matrices that are stored in a transposed layout.
    # I.e., tensor.t().contiguous() and we want the kernel to internally evaluate it as untransposed.
    # If we just take a view, like tensor.t(), the strides will account for it, so no flag is needed.
    is_a_transposed: bool = False
    is_b_transposed: bool = False
    shared_b: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    activation: Optional[Callable] = None

@triton.jit
def m_grouped_gemm_inner_kernel(
    # Tile ids
    problem_id,
    tile_id,
    start_idx,
    end_idx,
    last_problem_end,
    # Input parameters
    a_ptr,
    a_strides,
    b_ptr,
    b_strides,
    out_ptr,
    out_strides,
    group_indices_ptr,
    permute_indices_ptr,
    m,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    IS_A_TRANSPOSED: tl.constexpr,
    IS_B_TRANSPOSED: tl.constexpr,
    # Kernel parameters
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr,
    USE_TENSOR_DESCRIPTOR: tl.constexpr,
):          
    num_m_tiles = tl.cdiv(m, BLOCK_M)
    num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_N)    
    num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
    tl.assume(num_tiles >= 0)
    end_tile_id = last_problem_end + num_tiles
    tl.assume(end_tile_id >= tile_id)
    
    # TODO: Struggling to get this loop to flatten, so the pipeline has bubbles.
    # Checking ttgir, clearly the loops are not being fused and there is an async_wait
    # after the inner mma loop. Same output with either flatten=True/False
    for _ in tl.range(tile_id, end_tile_id, NUM_PROGRAMS, flatten=True):
        
        tile_id_in_gemm = tile_id - last_problem_end

        if CACHE_GROUP_M == 0:
            tile_m_idx = (tile_id_in_gemm // num_n_tiles) * BLOCK_M
            tile_n_idx = (tile_id_in_gemm % num_n_tiles) * BLOCK_N
        else:
            # On 3.4.0, working around several potential compiler bugs. 
            # 1. triton doesn't like multiplying the group size with the num_n_tiles. Going through another
            # variable, group_m, gets it to compile. It also doesn't work to manually inline a group size.
            # 2. trying to use tl.swizzle2d or putting into a func hits a "failures [...]
            # while processing an MLIR pass pipeline"
            group_m = CACHE_GROUP_M
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
        
        if GATHER_ROWS:
            # Can avoid masking, since oversets are 0 <= tile_m_offsets < m
            permute_a_indices = tl.load(permute_indices_ptr + start_idx + tile_m_offsets)
            a_indices = permute_a_indices // TOPK
        else:
            a_indices = start_idx + tile_m_offsets

        k_offset = tl.arange(0, BLOCK_K)

        if IS_A_TRANSPOSED:
            a_row_offsets = k_offset * a_strides[0]
            a_col_offsets = a_indices * a_strides[1]
        else:
            a_row_offsets = a_indices * a_strides[0]
            a_col_offsets = k_offset * a_strides[1]
        a_ptrs = a_ptr + a_row_offsets[:, None] + a_col_offsets

        b_problem_offset = problem_id * b_strides[0]
        if IS_B_TRANSPOSED:
            b_row_offsets = tile_n_offsets * b_strides[1]
            b_col_offsets = k_offset * b_strides[2]
        else:
            b_row_offsets = k_offset * b_strides[1]
            b_col_offsets = tile_n_offsets * b_strides[2]            
        b_ptrs = b_ptr + b_problem_offset + b_row_offsets[:, None] + b_col_offsets
        
        MASK_N: tl.constexpr = N % BLOCK_N != 0
        MASK_K: tl.constexpr = K % BLOCK_K != 0
        
        if MASK_N:
            n_mask = tile_n_offsets < N

        token_mask = start_idx + tile_m_offsets < end_idx

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b_ptrs, [16, 16])
            
            k_remaining = K - k * BLOCK_K
            if MASK_N and MASK_K:
                a_mask = token_mask[:, None] & (k_offset < k_remaining)
                b_mask = n_mask[None, :] & (k_offset[:, None] < k_remaining)
            elif MASK_K:
                a_mask = token_mask[:, None] & (k_offset < k_remaining)
                b_mask = k_offset[:, None] < k_remaining
            elif MASK_N:
                a_mask = token_mask[:, None]
                b_mask = n_mask[None, :]
            else:
                a_mask = token_mask[:, None]
             
            if IS_A_TRANSPOSED:
                a_mask = a_mask.T
            if IS_B_TRANSPOSED and (MASK_N or MASK_K):
                b_mask = b_mask.T

            # TODO: this branch may not be necessary if triton is able
            # to optimize away the masking on its own.
            if MASK_N or MASK_K:
                a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
            else:
                a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b_block = tl.load(b_ptrs)

            if IS_A_TRANSPOSED:
                a_block = a_block.T
            if IS_B_TRANSPOSED:
                b_block = b_block.T

            acc = tl.dot(a_block, b_block, acc=acc, input_precision="ieee")
            
            if IS_A_TRANSPOSED:
                a_ptrs += BLOCK_K * a_strides[0]
            else:
                a_ptrs += BLOCK_K * a_strides[1]        
            if IS_B_TRANSPOSED:
                b_ptrs += BLOCK_K * b_strides[2]
            else:
                b_ptrs += BLOCK_K * b_strides[1]

        # Splitting the epilogue is supposed to help overlap the next iteration 
        # of the outer loop with the epilogue.
        accs = epilogue_split(acc, EPILOGUE_SPLIT, EPILOGUE, BLOCK_M, BLOCK_N)

        tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
        tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets % m, BLOCK_M), BLOCK_M)
        a_mask = start_idx + tile_m_offsets < end_idx
        # The accumulators are all the same size, but the EPILOGUE may change the 
        # tile size in the N-dimension, so we use .shape[1], rather than BLOCK_N.
        out_tile_n_offsets = tile_n_idx // BLOCK_N * (accs[0].shape[1] * EPILOGUE_SPLIT) + tl.arange(0, accs[0].shape[1])
        out_tile_n_offsets = tl.max_contiguous(tl.multiple_of(out_tile_n_offsets, accs[0].shape[1]), accs[0].shape[1])

        if SCATTER_ROWS:    
            # Can avoid masking, since offsets are 0 <= tile_m_offsets < m
            permute_a_indices = tl.load(permute_indices_ptr + start_idx + tile_m_offsets)     
            out_offsets = permute_a_indices[:, None] * out_strides[0] + out_tile_n_offsets * out_strides[1]
            out_ptrs = out_ptr + out_offsets
        else:
            out_row_offsets = start_idx + tile_m_offsets
            out_offsets = out_row_offsets[:, None] * out_strides[0] + out_tile_n_offsets * out_strides[1]
            out_ptrs = out_ptr + out_offsets

        store_split_epilogue(out_ptrs, out_strides[1], a_mask, N, accs)

        tile_id += NUM_PROGRAMS
    
    return tile_id, num_tiles

@triton.jit
def m_grouped_gemm_persistent_kernel(
    a_ptr,
    a_strides,
    b_ptr,
    b_strides,
    out_ptr,
    out_strides,
    group_indices_ptr,
    permute_indices_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    IS_A_TRANSPOSED: tl.constexpr,
    IS_B_TRANSPOSED: tl.constexpr,
    # Kernel parameters
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr,
    USE_TENSOR_DESCRIPTOR: tl.constexpr
):
    tile_id = tl.program_id(axis=0)
    last_problem_end = 0

    tl.assume(tile_id >= 0)
    tl.assume(a_strides[0] > 0)
    tl.assume(a_strides[1] > 0)
    tl.assume(b_strides[0] > 0)
    tl.assume(b_strides[1] > 0)
    tl.assume(b_strides[2] > 0)
    tl.assume(out_strides[0] > 0)
    tl.assume(out_strides[1] > 0)
    
    start_idx = 0
    for problem_id in tl.range(0, NUM_EXPERTS):
        end_idx = tl.load(group_indices_ptr + problem_id + 1, cache_modifier=".ca")
        m = end_idx - start_idx
        
        tl.assume(start_idx >= 0)
        tl.assume(end_idx >= start_idx)
        tl.assume(m >= 0)
        tl.assume(m <= NUM_TOKENS * TOPK)
        MASK_N: tl.constexpr = N % BLOCK_N != 0
        MASK_K: tl.constexpr = K % BLOCK_K != 0
                
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_N)    
        num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
        tl.assume(num_tiles >= 0)
        end_tile_id = last_problem_end + num_tiles
        tl.assume(end_tile_id >= tile_id)
        
        tile_id, num_tiles = m_grouped_gemm_inner_kernel(
            problem_id,
            tile_id,
            start_idx,
            end_idx,
            last_problem_end,
            a_ptr,
            a_strides,
            b_ptr,
            b_strides,
            out_ptr,
            out_strides,
            group_indices_ptr,
            permute_indices_ptr,
            m,
            K,
            N,
            TOPK,
            GATHER_ROWS,
            SCATTER_ROWS,
            IS_A_TRANSPOSED,
            IS_B_TRANSPOSED,
            NUM_PROGRAMS,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            CACHE_GROUP_M,
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

def m_grouped_gemm_default_config(e, params, dtype):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    num_stages = 5
    if dtype == torch.float32:
        BLOCK_N //= 2
        num_stages -= 1
    if not (params.gather or params.scatter):
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "CACHE_GROUP_M": 8,
                "EPILOGUE_SPLIT": 2,
                "DISALLOW_ACC_MULTI_BUFFER": False,
                "USE_TENSOR_DESCRIPTOR": False,
            },
            num_warps=8, 
            num_stages=num_stages
        )
    elif params.gather:
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "CACHE_GROUP_M": 8,
                "EPILOGUE_SPLIT": 2,
                "DISALLOW_ACC_MULTI_BUFFER": False,
                "USE_TENSOR_DESCRIPTOR": False,
            },
            num_warps=8, 
            num_stages=num_stages
        )
    elif params.scatter:
        default_config = triton.Config({
                "BLOCK_M": BLOCK_M, 
                "BLOCK_N": BLOCK_N, 
                "BLOCK_K": BLOCK_K, 
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "CACHE_GROUP_M": 8,
                "EPILOGUE_SPLIT": 2,
                "DISALLOW_ACC_MULTI_BUFFER": False,
                "USE_TENSOR_DESCRIPTOR": False,
            },
            num_warps=8, 
            num_stages=num_stages
        )
    return default_config

def m_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_indices: torch.Tensor,
    params: MGroupedGEMMParams,
    autotune_mode: Optional[AutotuneMode] = None
):
    assert a.dim() == 2
    assert b.dim() == 3
    assert autotune_mode is None or autotune_mode in AutotuneMode

    if params.gather or params.scatter:
        assert params.permute_indices is not None

    num_tokens = params.num_tokens
    e, k, n = b.size()

    if params.gather or params.scatter:
        out_rows = num_tokens * params.topk
    else:
        out_rows = num_tokens
        
    out_cols = n
    if params.activation is not None and "glu" in params.activation:
        out_cols //= 2

    out = torch.empty((out_rows, out_cols), device=a.device, dtype=a.dtype)

    default_config = m_grouped_gemm_default_config(b.size(0), params, a.dtype)
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
        a, 
        a.stride(),
        b,
        b.stride(),
        out,
        out.stride(),
        group_indices, 
        params.permute_indices, 
        NUM_TOKENS=num_tokens,
        NUM_EXPERTS=e, 
        K=k,
        N=n,
        TOPK=params.topk,
        GATHER_ROWS=params.gather,
        SCATTER_ROWS=params.scatter,
        IS_A_TRANSPOSED=params.is_a_transposed,
        IS_B_TRANSPOSED=params.is_b_transposed,
        EPILOGUE=epilogue,
        **default_kwargs
    )
    
    return out