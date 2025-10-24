from dataclasses import dataclass
from functools import partial
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Optional, Callable
from moe_explore.gpu_utils import get_gpu_sm_count
from .activation import TRITON_ACTIVATIONS, activation
from .autotune_config import (
    AutotuneMode,
    fast_autotune_configs, 
    max_autotune_configs
)
from .epilogue_split import epilogue_split, store_split_epilogue
from .tile_util import get_tile_id_in_group, tile_offsets_in_group

@dataclass
class KGroupedGEMMParams:
    permute_indices: Optional[torch.Tensor]
    gather_a: bool
    gather_b: bool
    # `num_tokens` is confusing. When doing a scatter or gather, it's the number of tokens
    # before routing! If we are NOT doing a scatter or gather, it's just the number of tokens being input.
    # If we do a gather, `num_tokens` is the number of tokens in the input, before the gather.
    # Similar for when we do a scatter, it's the number of tokens in the output, after the scatter.
    num_tokens: int
    topk: int
    activation: Optional[Callable] = None

@triton.jit
def k_grouped_gemm_inner_kernel(
    # Tile ids
    problem_id,
    tile_id,
    start_idx,
    end_idx,
    last_problem_end,
    # Input parameters
    a_ptr,
    a_stride_1, a_stride_2,
    b_ptr,
    b_stride_1, b_stride_2,
    out_ptr,
    out_stride_1, out_stride_2, out_stride_3,
    group_indices_ptr,
    permute_indices_ptr,
    M: tl.constexpr,
    k,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_A: tl.constexpr,
    GATHER_B: tl.constexpr,
    USE_OUT_TENSOR_DESCRIPTOR: tl.constexpr,
    # Kernel parameters
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
):
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)    
    num_tiles = num_m_tiles * num_n_tiles
    end_tile_id = last_problem_end + num_tiles
    tl.assume(num_tiles >= 0)
    tl.assume(end_tile_id >= tile_id)
    
    # TODO: Struggling to get this loop to flatten, so the pipeline has bubbles.
    # Checking ttgir, clearly the loops are not being fused and there is an async_wait
    # after the inner mma loop. Same output with either flatten=True/False.
    for _ in tl.range(tile_id, end_tile_id, NUM_PROGRAMS, flatten=True):
        
        tile_id_in_gemm = tile_id - last_problem_end

        tile_m_idx, tile_n_idx = get_tile_id_in_group(
            tile_id_in_gemm, M, N, BLOCK_M, BLOCK_N, CACHE_GROUP_M)
        tile_m_offsets, tile_n_offsets = tile_offsets_in_group(
            tile_m_idx, tile_n_idx, M, N, BLOCK_M, BLOCK_N)
        
        k_offset = tl.arange(0, BLOCK_K)
        
        if not GATHER_A:
            a_row_offsets = (start_idx + k_offset) * a_stride_1
            a_col_offsets = tile_m_offsets * a_stride_2
            a_ptrs = a_ptr + a_row_offsets[:, None] + a_col_offsets

        if not GATHER_B:
            b_row_offsets = (start_idx + k_offset) * b_stride_1
            b_col_offsets = tile_n_offsets * b_stride_2            
            b_ptrs = b_ptr + b_row_offsets[:, None] + b_col_offsets
       
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_iter in tl.range(0, tl.cdiv(k, BLOCK_K)):
            #if not GATHER_A:
            #    tl.multiple_of(a_ptrs, [16, 16])
            #if not GATHER_B:
            #    tl.multiple_of(b_ptrs, [16, 16])
            
            k_step_offset = start_idx + tl.arange(0, BLOCK_K) + k_iter * BLOCK_K
            k_mask = k_step_offset < end_idx
            
            a_mask = k_mask[:, None] & (tile_m_offsets < M)
            b_mask = k_mask[:, None] & (tile_n_offsets < N)
        
            if GATHER_A or GATHER_B:
                # Ideallly, thsi will be pipeline as well...!
                gather_indices = tl.load(permute_indices_ptr + k_step_offset,
                                         mask=k_mask,
                                         other=0)
                
            if GATHER_A:
                a_row_offsets = (gather_indices // TOPK) * a_stride_1
                a_col_offsets = tile_m_offsets * a_stride_2
                a_ptrs = a_ptr + a_row_offsets[:, None] + a_col_offsets
                tl.multiple_of(a_ptrs, [16, 16])
            if GATHER_B:
                b_row_offsets = gather_indices * b_stride_1
                b_col_offsets = tile_n_offsets * b_stride_2
                b_ptrs = b_ptr + b_row_offsets[:, None] + b_col_offsets
                tl.multiple_of(b_ptrs, [16, 16])

            a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            acc = tl.dot(a_block.T, b_block, acc=acc, input_precision="ieee")
           
            if not GATHER_A:
                a_ptrs += BLOCK_K * a_stride_1
            if not GATHER_B:
                b_ptrs += BLOCK_K * b_stride_1

        # Splitting the epilogue is supposed to help overlap the next iteration 
        # of the outer loop with the epilogue.
        accs = epilogue_split(acc, EPILOGUE_SPLIT, BLOCK_M, BLOCK_N)

        if not USE_OUT_TENSOR_DESCRIPTOR:
            tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
            tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets % M, BLOCK_M), BLOCK_M)
            out_m_mask = tile_m_idx + tl.arange(0, BLOCK_M) < M
            out_tile_n_offsets = tile_n_idx + tl.arange(0, accs[0].shape[1]) #// BLOCK_N * (accs[0].shape[1] * EPILOGUE_SPLIT) + tl.arange(0, accs[0].shape[1])
            #out_tile_n_offsets = tl.max_contiguous(tl.multiple_of(out_tile_n_offsets, accs[0].shape[1]), accs[0].shape[1])
            out_row_offsets = tile_m_offsets
            out_offsets = problem_id * out_stride_1 + out_row_offsets[:, None] * out_stride_2 + out_tile_n_offsets * out_stride_3
            out_ptrs = out_ptr + out_offsets
            for i in tl.static_range(len(accs)):
                out = accs[i]
                n_offset = tl.arange(0, out.shape[1])
                epilogue_split_offset = i * out.shape[1]
                tl.store(
                    out_ptrs + epilogue_split_offset * out_stride_3, 
                    out,
                    mask=out_m_mask[:, None] & (epilogue_split_offset + n_offset < N - tile_n_idx))
        else:
            for i in tl.static_range(len(accs)):
                # need to explicitly cast `out` dtype to match the descriptor dtype
                out = accs[i].expand_dims(0).to(out_ptr.dtype)
                out_ptr.store([problem_id, tile_m_idx, tile_n_idx + i * out.shape[1]], out)
            
        tile_id += NUM_PROGRAMS
    
    return tile_id, num_tiles

@triton.jit
def k_grouped_gemm_persistent_kernel(
    a_ptr,
    a_stride_1, a_stride_2,
    b_ptr,
    b_stride_1, b_stride_2,
    out_ptr,
    out_stride_1, out_stride_2, out_stride_3,
    group_indices_ptr,
    permute_indices_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_A: tl.constexpr,
    GATHER_B: tl.constexpr,
    USE_OUT_TENSOR_DESCRIPTOR: tl.constexpr,
    # Kernel parameters
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    last_problem_end = 0

    tl.assume(tile_id >= 0)
    tl.assume(a_stride_1 > 0)
    tl.assume(a_stride_2 > 0)
    tl.assume(b_stride_1 > 0)
    tl.assume(b_stride_2 > 0)
    tl.assume(out_stride_1 > 0)
    tl.assume(out_stride_2 > 0)
    tl.assume(out_stride_3 > 0)
    
    for problem_id in tl.range(0, NUM_EXPERTS):
        group_bounds = tl.load(group_indices_ptr + problem_id + tl.arange(0, 2), cache_modifier=".ca")
        start_idx, end_idx = group_bounds.split()
        k = end_idx - start_idx
        
        tl.assume(start_idx >= 0)
        tl.assume(end_idx >= start_idx)
        tl.assume(k >= 0)
        tl.assume(k <= NUM_TOKENS * TOPK)
        
        tile_id, num_tiles = k_grouped_gemm_inner_kernel(
            problem_id,
            tile_id,
            start_idx,
            end_idx,
            last_problem_end,
            a_ptr,
            a_stride_1, a_stride_2,
            b_ptr,
            b_stride_1, b_stride_2,
            out_ptr,
            out_stride_1, out_stride_2, out_stride_3,
            group_indices_ptr,
            permute_indices_ptr,
            M,
            k,
            N,
            TOPK,
            GATHER_A,
            GATHER_B,
            USE_OUT_TENSOR_DESCRIPTOR,
            NUM_PROGRAMS,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            CACHE_GROUP_M,
            EPILOGUE_SPLIT,
        )
        
        last_problem_end += num_tiles

_fast_autotune_k_grouped_gemm_persistent_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=['NUM_TOKENS', 'E', 'M', 'N', 'GATHER_A', 'GATHER_B'],
    reset_to_zero=['out_ptr']
)(k_grouped_gemm_persistent_kernel)

_max_autotune_k_grouped_gemm_persistent_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True),
    key=['NUM_TOKENS', 'E', 'M', 'N', 'GATHER_A', 'GATHER_B'],
    reset_to_zero=['out_ptr']
)(k_grouped_gemm_persistent_kernel)

def k_grouped_gemm_default_config(e, params, dtype):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    num_stages = 5
    if dtype == torch.float32:
        #BLOCK_N //= 2
        num_stages = 3
    default_config = triton.Config({
            "BLOCK_M": BLOCK_M, 
            "BLOCK_N": BLOCK_N, 
            "BLOCK_K": BLOCK_K,
            "NUM_PROGRAMS": get_gpu_sm_count(),
            "CACHE_GROUP_M": 8,
            "EPILOGUE_SPLIT": 2,
            "USE_OUT_TENSOR_DESCRIPTOR": False
        },
        num_warps=8, 
        num_stages=num_stages
    )
    return default_config

def k_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_indices: torch.Tensor,
    params: KGroupedGEMMParams,
    autotune_mode: Optional[AutotuneMode] = None
):
    assert a.dim() == 2
    assert b.dim() == 2
    assert autotune_mode is None or autotune_mode in AutotuneMode

    if params.gather_a or params.gather_b:
        assert params.permute_indices is not None

    num_tokens = params.num_tokens
    _, m = a.size()
    _, n = b.size()

    out = torch.empty((group_indices.size(0) - 1, m, n), device=a.device, dtype=a.dtype)

    default_config = k_grouped_gemm_default_config(group_indices.size(0) - 1, params, a.dtype)
    default_kwargs = default_config.all_kwargs()
    del default_kwargs["num_ctas"]
    
    block_m, block_n = default_kwargs["BLOCK_M"], default_kwargs["BLOCK_N"]
    use_out_tensor_descriptor = default_kwargs["USE_OUT_TENSOR_DESCRIPTOR"]
    del default_kwargs["USE_OUT_TENSOR_DESCRIPTOR"]
    # only use descriptors when strides, except the last, are 16-byte aligned
    use_out_tensor_descriptor = use_out_tensor_descriptor and all([(s * out.element_size()) % 16 == 0 for s in out.stride()[:-1]])
    # TODO: I'm not sure why it doesn't work with float32...
    use_out_tensor_descriptor = use_out_tensor_descriptor and out.dtype in (torch.float16, torch.bfloat16)
    if use_out_tensor_descriptor:
        # the descriptor needs to account for the potential epilogue splitting.
        block_size = [1, block_m, block_n // default_kwargs["EPILOGUE_SPLIT"]]
        out_desc = TensorDescriptor.from_tensor(out, block_size)

    func = k_grouped_gemm_persistent_kernel
    if autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_k_grouped_gemm_persistent_kernel
        default_kwargs = {}
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_k_grouped_gemm_persistent_kernel
        default_kwargs = {}

    grid = lambda META: (META["NUM_PROGRAMS"],)
        
    func[grid](
        a, 
        a.stride(0), a.stride(1),
        b,
        b.stride(0), b.stride(1),
        out_desc if use_out_tensor_descriptor else out,
        out.stride(0), out.stride(1), out.stride(2),
        group_indices, 
        params.permute_indices, 
        NUM_TOKENS=num_tokens,
        NUM_EXPERTS=group_indices.size(0) - 1, 
        M=m,
        N=n,
        TOPK=params.topk,
        GATHER_A=params.gather_a,
        GATHER_B=params.gather_b,
        USE_OUT_TENSOR_DESCRIPTOR=False,#use_out_tensor_descriptor,
        **default_kwargs
    )
    
    return out