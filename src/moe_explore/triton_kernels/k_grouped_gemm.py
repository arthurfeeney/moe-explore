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
    K_LOOP_STAGES: tl.constexpr
):
    tl.assume(a_stride_1 > 0)
    tl.assume(a_stride_2 > 0)
    tl.assume(b_stride_1 > 0)
    tl.assume(b_stride_2 > 0)
    tl.assume(out_stride_1 > 0)
    tl.assume(out_stride_2 > 0)
    tl.assume(out_stride_3 > 0)
    
    MASK_M: tl.constexpr = M % BLOCK_M != 0
    MASK_N: tl.constexpr = N % BLOCK_N != 0
    
    NUM_M_TILES = tl.cdiv(M, BLOCK_M)
    NUM_N_TILES = tl.cdiv(N, BLOCK_N)
    TILES_PER_GROUP = NUM_M_TILES * NUM_N_TILES
    for tile_id in tl.range(tl.program_id(0), NUM_EXPERTS * TILES_PER_GROUP, NUM_PROGRAMS, flatten=True):
        
        group_id = tile_id // TILES_PER_GROUP
        tile_id_in_group = tile_id - (group_id * TILES_PER_GROUP)
        
        tile_m_idx, tile_n_idx = get_tile_id_in_group(
            tile_id_in_group, M, N, BLOCK_M, BLOCK_N, CACHE_GROUP_M)
        tile_m_offsets, tile_n_offsets = tile_offsets_in_group(
            tile_m_idx, tile_n_idx, M, N, BLOCK_M, BLOCK_N)
        tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets, BLOCK_M), BLOCK_M)
        tile_n_offsets = tl.max_contiguous(tl.multiple_of(tile_n_offsets, BLOCK_N), BLOCK_N)
        
        if group_id == 0:
            start_idx = 0
            end_idx = tl.load(group_indices_ptr + group_id)
        else:
            group_bounds = tl.load(group_indices_ptr + group_id + tl.arange(0, 2) - 1)
            start_idx, end_idx = group_bounds.split()

        k = end_idx - start_idx
        k_offset = tl.arange(0, BLOCK_K)
    
        if not GATHER_A:
            a_row_offsets = (start_idx + k_offset) * a_stride_1
            a_col_offsets = tile_m_offsets * a_stride_2
            #tl.multiple_of(a_row_offsets, [16])
            #tl.multiple_of(a_col_offsets, [16])
            # load transposed?
            a_ptrs = a_ptr + a_row_offsets[:, None] + a_col_offsets

        if not GATHER_B:
            b_row_offsets = (start_idx + k_offset) * b_stride_1
            b_col_offsets = tile_n_offsets * b_stride_2            
            b_ptrs = b_ptr + b_row_offsets[:, None] + b_col_offsets
        
        if GATHER_A or GATHER_B:
            permute_indices_offsets = start_idx + tl.arange(0, BLOCK_K)
            permute_indices_offsets = tl.max_contiguous(tl.multiple_of(permute_indices_offsets, BLOCK_K), BLOCK_K)
            permute_indices_ptrs = permute_indices_ptr + permute_indices_offsets
            
        if GATHER_A:
            a_col_ptrs = a_ptr + tile_m_offsets * a_stride_2
    
        if GATHER_B:
            b_col_ptrs = b_ptr + tile_n_offsets * b_stride_2
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_iter in tl.range(0, tl.cdiv(k, BLOCK_K), num_stages=K_LOOP_STAGES):
            if not GATHER_A:
                tl.multiple_of(a_ptrs, [16, 16])
            if not GATHER_B:
                tl.multiple_of(b_ptrs, [16, 16])
            if GATHER_A or GATHER_B:
                tl.multiple_of(permute_indices_ptrs, [16])
            if GATHER_A:
                tl.multiple_of(a_col_ptrs, [16])
            if GATHER_B:
                tl.multiple_of(b_col_ptrs, [16])
            
            if GATHER_A or GATHER_B:
                gather_indices = tl.load(permute_indices_ptrs)
                #gather_indices = tl.max_contiguous(tl.multiple_of(gather_indices, BLOCK_K), BLOCK_K)
                
            if GATHER_A:
                a_row_offsets = (gather_indices // TOPK) * a_stride_1
                #a_row_offsets = tl.max_contiguous(tl.multiple_of(a_row_offsets, BLOCK_K), BLOCK_K)
                a_ptrs = a_col_ptrs + a_row_offsets[:, None]
                tl.multiple_of(a_ptrs, [16, 16])
            if GATHER_B:
                b_row_offsets = gather_indices * b_stride_1
                b_ptrs = b_col_ptrs + b_row_offsets[:, None]
                tl.multiple_of(b_ptrs, [16, 16])
            
            k_remaining = k - k_iter * BLOCK_K
            k_mask = k_offset < k_remaining

            if MASK_M:
                a_mask = k_mask[:, None] & (tile_m_offsets < M)
            else:
                a_mask = k_mask[:, None]
            
            if MASK_N:
                b_mask = k_mask[:, None] & (tile_n_offsets < N)
            else:
                b_mask = k_mask[:, None]

            a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            acc = tl.dot(a_block.T, b_block, acc=acc, input_precision="ieee")
           
            if not GATHER_A:
                a_ptrs += BLOCK_K * a_stride_1
            if not GATHER_B:
                b_ptrs += BLOCK_K * b_stride_1
            if GATHER_A or GATHER_B:
                permute_indices_ptrs += BLOCK_K
        
        # Splitting the epilogue is supposed to help overlap the next iteration 
        # of the outer loop with the epilogue.
        accs = epilogue_split(acc, EPILOGUE_SPLIT, BLOCK_M, BLOCK_N)
        
        if not USE_OUT_TENSOR_DESCRIPTOR:
            #tile_id_in_gemm = tile_id - group_id * TILES_PER_GROUP
            #tile_m_idx, tile_n_idx = get_tile_id_in_group(
            #    tile_id_in_gemm, M, N, BLOCK_M, BLOCK_N, CACHE_GROUP_M)
            #tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
            #tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets % M, BLOCK_M), BLOCK_M)
            out_m_mask = tile_m_idx + tl.arange(0, BLOCK_M) < M
            out_tile_n_offsets = tile_n_idx + tl.arange(0, accs[0].shape[1]) #// BLOCK_N * (accs[0].shape[1] * EPILOGUE_SPLIT) + tl.arange(0, accs[0].shape[1])
            out_tile_n_offsets = tl.max_contiguous(tl.multiple_of(out_tile_n_offsets, accs[0].shape[1]), accs[0].shape[1])
            out_row_offsets = tile_m_offsets
            out_offsets = group_id * out_stride_1 + out_row_offsets[:, None] * out_stride_2 + out_tile_n_offsets * out_stride_3
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
                out_ptr.store([group_id, tile_m_idx, tile_n_idx + i * out.shape[1]], out)

_fast_autotune_k_grouped_gemm_persistent_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=['NUM_TOKENS', 'E', 'M', 'N', 'GATHER_A', 'GATHER_B'],
    reset_to_zero=['out_ptr']
)(k_grouped_gemm_persistent_kernel)

_max_autotune_k_grouped_gemm_persistent_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True, k_loop_stages=True),
    key=['NUM_TOKENS', 'E', 'M', 'N', 'GATHER_A', 'GATHER_B'],
    reset_to_zero=['out_ptr']
)(k_grouped_gemm_persistent_kernel)

def k_grouped_gemm_default_config(e, params, dtype):
    #BLOCK_M = 128
    #LOCK_N = 256
    #BLOCK_K = 32
    #num_stages = 3
    #if dtype == torch.float32:
        #BLOCK_N //= 2
    #    num_stages = 3
    default_config = triton.Config({
            "BLOCK_M": 128, 
            "BLOCK_N": 128, 
            "BLOCK_K": 64,
            "NUM_PROGRAMS": get_gpu_sm_count(),
            "CACHE_GROUP_M": 8,
            "EPILOGUE_SPLIT": 2,
            "USE_OUT_TENSOR_DESCRIPTOR": False,
            "K_LOOP_STAGES": 5
        },
        num_warps=8,
        num_stages=3
    )
    if params.gather_a:
        default_config = triton.Config({
                "BLOCK_M": 128, 
                "BLOCK_N": 128, 
                "BLOCK_K": 32,
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "CACHE_GROUP_M": 8,
                "EPILOGUE_SPLIT": 2,
                "USE_OUT_TENSOR_DESCRIPTOR": False,
                "K_LOOP_STAGES": 5
            },
            num_warps=8,
            num_stages=5
        )
    elif params.gather_b:
        default_config = triton.Config({
                "BLOCK_M": 128, 
                "BLOCK_N": 128, 
                "BLOCK_K": 32,
                "NUM_PROGRAMS": get_gpu_sm_count(),
                "CACHE_GROUP_M": 8,
                "EPILOGUE_SPLIT": 2,
                "USE_OUT_TENSOR_DESCRIPTOR": False,
                "K_LOOP_STAGES": 5
            },
            num_warps=8,
            num_stages=5
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

    out = torch.empty((group_indices.size(0), m, n), device=a.device, dtype=a.dtype)

    default_config = k_grouped_gemm_default_config(group_indices.size(0), params, a.dtype)
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
        NUM_EXPERTS=group_indices.size(0), 
        M=m,
        N=n,
        TOPK=params.topk,
        GATHER_A=params.gather_a,
        GATHER_B=params.gather_b,
        USE_OUT_TENSOR_DESCRIPTOR=False,#use_out_tensor_descriptor,
        **default_kwargs
    )
    
    return out