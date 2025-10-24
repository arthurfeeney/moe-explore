from dataclasses import dataclass
from functools import partial
import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Optional, Callable
from moe_explore.gpu_utils import get_gpu_sm_count
from moe_explore.triton_kernels.tile_util import get_tile_id_in_group, tile_offsets_in_group
from moe_explore.triton_kernels.activation import activation, act_n
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
    return_preactivation: bool = False
    shared_b: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    activation: Optional[Callable] = None
    # This is the pre-activation from a forward pass. It's
    # used to compute the fused activation in a backward pass.
    pre_act_for_grad: Optional[torch.Tensor] = None

@dataclass
class MGroupedGEMMOutput:
    output: torch.Tensor
    preactivation: Optional[torch.Tensor] = None

@triton.jit
def m_grouped_gemm_inner_kernel(
    # Tile ids
    problem_id,
    tile_id,
    end_tile_id,
    start_idx,
    end_idx,
    last_problem_end,
    # Input parameters,
    a_ptr,
    a_stride_1, a_stride_2,
    b_ptr,
    b_stride_1, b_stride_2, b_stride_3,
    out_ptr,
    out_stride_1, out_stride_2,
    preactivation_ptr,
    preactivation_stride_1, preactivation_stride_2,
    preact_for_grad_ptr,
    preact_for_grad_stride_1, preact_for_grad_stride_2,
    permute_indices_ptr,
    m,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    RETURN_PREACTIVATION: tl.constexpr,
    IS_A_TRANSPOSED: tl.constexpr,
    IS_B_TRANSPOSED: tl.constexpr,
    USE_A_TENSOR_DESCRIPTOR: tl.constexpr,
    USE_B_TENSOR_DESCRIPTOR: tl.constexpr,
    # Kernel parameters
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
    GRAD_ACT: tl.constexpr
):  
    
    if USE_A_TENSOR_DESCRIPTOR:
        a_desc = tl.make_tensor_descriptor(
            a_ptr + start_idx * K,
            shape=(m, K),
            strides=(K, 1), # Assuming row major order
            block_shape=(BLOCK_M, BLOCK_K)
        )
    
    # TODO: Struggling to get this loop to flatten, so the pipeline has bubbles.
    # Checking ttgir, clearly the loops are not being fused and there is an async_wait
    # after the inner mma loop. Same output with either flatten=True/False
    for _ in tl.range(tile_id, end_tile_id, NUM_PROGRAMS, flatten=True):
        
        tile_id_in_gemm = tile_id - last_problem_end
        tile_m_idx, tile_n_idx = get_tile_id_in_group(
            tile_id_in_gemm, m, N, BLOCK_M, BLOCK_N, CACHE_GROUP_M)
        tile_m_offsets, tile_n_offsets = tile_offsets_in_group(
            tile_m_idx, tile_n_idx, m, N, BLOCK_M, BLOCK_N)
        
        if GATHER_ROWS:
            permute_a_indices = tl.load(permute_indices_ptr + start_idx + tile_m_offsets)
            a_indices = permute_a_indices // TOPK
        else:
            a_indices = start_idx + tile_m_offsets

        k_offset = tl.arange(0, BLOCK_K)

        if IS_A_TRANSPOSED:
            a_row_offsets = k_offset * a_stride_1
            a_col_offsets = a_indices * a_stride_2
        else:
            a_row_offsets = a_indices * a_stride_1
            a_col_offsets = k_offset * a_stride_2
        a_ptrs = a_ptr + a_row_offsets[:, None] + a_col_offsets

        if not USE_B_TENSOR_DESCRIPTOR:
            b_problem_offset = problem_id * b_stride_1
            if IS_B_TRANSPOSED:
                b_row_offsets = tile_n_offsets * b_stride_2
                b_col_offsets = k_offset * b_stride_3
            else:
                b_row_offsets = k_offset * b_stride_2
                b_col_offsets = tile_n_offsets * b_stride_3            
            b_ptrs = b_ptr + b_problem_offset + b_row_offsets[:, None] + b_col_offsets

        n_mask = tile_n_offsets < N
        token_mask = start_idx + tile_m_idx + tl.arange(0, BLOCK_M) < end_idx

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            if not USE_A_TENSOR_DESCRIPTOR:
                tl.multiple_of(a_ptrs, [16, 16])
            if not USE_B_TENSOR_DESCRIPTOR:
                tl.multiple_of(b_ptrs, [16, 16])

            k_remaining = K - k * BLOCK_K
            if not USE_A_TENSOR_DESCRIPTOR:
                a_mask = token_mask[:, None] & (k_offset < k_remaining)
                if IS_A_TRANSPOSED:
                    a_mask = a_mask.T
            if not USE_B_TENSOR_DESCRIPTOR:
                b_mask = n_mask[None, :] & (k_offset[:, None] < k_remaining)
                if IS_B_TRANSPOSED:
                    b_mask = b_mask.T

            if USE_A_TENSOR_DESCRIPTOR:
                a_block = a_desc.load([tile_m_idx, k * BLOCK_K])
            else:
                a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
            if USE_B_TENSOR_DESCRIPTOR:
                # TODO: .reshape is used because we're loading 3d [1, block1, block2] tiles
                # this could be removed by taking a 2d view, but I'm pretty sure this
                # is a no-op?
                if not IS_B_TRANSPOSED:
                    b_block = b_ptr.load([problem_id, k * BLOCK_K, tile_n_idx])
                    b_block = tl.reshape(b_block, (BLOCK_K, BLOCK_N))
                else:
                    b_block = b_ptr.load([problem_id, tile_n_idx, k * BLOCK_K])
                    b_block = tl.reshape(b_block, (BLOCK_N, BLOCK_K))    
            else:
                b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

            if IS_A_TRANSPOSED:
                a_block = a_block.T
            if IS_B_TRANSPOSED:
                b_block = b_block.T

            acc = tl.dot(a_block, b_block, acc=acc, input_precision="ieee")
            
            if IS_A_TRANSPOSED:
                a_ptrs += BLOCK_K * a_stride_1
            else:
                a_ptrs += BLOCK_K * a_stride_2    
                
            if not USE_B_TENSOR_DESCRIPTOR:    
                if IS_B_TRANSPOSED:
                    b_ptrs += BLOCK_K * b_stride_3
                else:
                    b_ptrs += BLOCK_K * b_stride_2

        accs = epilogue_split(acc, EPILOGUE_SPLIT, BLOCK_M, BLOCK_N)
        
        tile_id_in_gemm = tile_id - last_problem_end
        tile_m_idx, tile_n_idx = get_tile_id_in_group(
            tile_id_in_gemm, m, N, BLOCK_M, BLOCK_N, CACHE_GROUP_M)
        tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
        tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets, BLOCK_M), BLOCK_M)
        a_mask = start_idx + tile_m_offsets < end_idx

        if SCATTER_ROWS:
            tile_m_indices = tl.load(permute_indices_ptr + start_idx + tile_m_offsets)     
        else:
            tile_m_indices = start_idx + tile_m_offsets

        # These are the N-dim of the activation blocks
        PRE_ACT_SPLIT_N: tl.constexpr = BLOCK_N // EPILOGUE_SPLIT

        for i in tl.static_range(len(accs)):
            pre = accs[i]

            if RETURN_PREACTIVATION:
                tl.static_assert(not SCATTER_ROWS)
                tl.static_assert(pre.shape[1] == PRE_ACT_SPLIT_N)
                pre_tile_n_offsets = tile_n_idx + tl.arange(0, PRE_ACT_SPLIT_N)
                #pre_tile_n_offsets = tl.max_contiguous(tl.multiple_of(pre_tile_n_offsets, BLOCK_N), BLOCK_N)
                pre_ptrs = preactivation_ptr + tile_m_indices[:, None] * preactivation_stride_1 + pre_tile_n_offsets * preactivation_stride_2
                n_offset = tl.arange(0, PRE_ACT_SPLIT_N)
                epilogue_split_offset = i * PRE_ACT_SPLIT_N
                tl.store(
                    pre_ptrs + epilogue_split_offset * preactivation_stride_2, 
                    pre, 
                    mask=token_mask[:, None] & (tile_n_idx + epilogue_split_offset + n_offset < N))

            out = activation(pre, EPILOGUE, None) if EPILOGUE is not None else pre

            out_tile_n_idx = tile_n_idx // BLOCK_N * (out.shape[1] * EPILOGUE_SPLIT)
            out_tile_n_offsets = out_tile_n_idx + tl.arange(0, out.shape[1])
            out_ptrs = out_ptr + tile_m_indices[:, None] * out_stride_1 + out_tile_n_offsets * out_stride_2
            n_offset = tl.arange(0, out.shape[1])
            epilogue_split_offset = i * out.shape[1]
            
            # output n-dimension depends on the activation function
            if BLOCK_N > (out.shape[1] * EPILOGUE_SPLIT):
                OUT_N = N // 2
            else:
                OUT_N = N
            
            tl.store(
                out_ptrs + epilogue_split_offset * out_stride_2, 
                out,
                mask=token_mask[:, None] & (out_tile_n_idx + epilogue_split_offset + n_offset < OUT_N))
        
        tile_id += NUM_PROGRAMS
    return tile_id
    

@triton.jit
def m_grouped_gemm_persistent_kernel(
    a_ptr,
    a_stride_1, a_stride_2,
    b_ptr,
    b_stride_1, b_stride_2, b_stride_3,
    out_ptr,
    out_stride_1, out_stride_2,
    # This is an optional pointer to return the gemm before a fused activation
    preactivation_ptr,
    preactivation_stride_1, preactivation_stride_2,
    # This is an optional pointer to a pre-activation from a forward pass, it's
    # necessary for fusing the grad-activation in a backward pass.
    preact_for_grad_ptr,
    preact_for_grad_stride_1, preact_for_grad_stride_2,
    group_indices_ptr,
    permute_indices_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    SCATTER_ROWS: tl.constexpr,
    RETURN_PREACTIVATION: tl.constexpr,
    IS_A_TRANSPOSED: tl.constexpr,
    IS_B_TRANSPOSED: tl.constexpr,
    USE_A_TENSOR_DESCRIPTOR: tl.constexpr,
    USE_B_TENSOR_DESCRIPTOR: tl.constexpr,
    # Kernel parameters
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EPILOGUE_SPLIT: tl.constexpr,
    GRAD_ACT: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    last_problem_end = 0

    tl.assume(tile_id >= 0)
    tl.assume(a_stride_1 > 0)
    tl.assume(a_stride_2 > 0)
    tl.assume(b_stride_1 > 0)
    tl.assume(b_stride_2 > 0)
    tl.assume(b_stride_3 > 0)
    tl.assume(out_stride_1 > 0)
    tl.assume(out_stride_2 > 0)
    
    for problem_id in tl.range(0, NUM_EXPERTS):
        group_indices = tl.load(group_indices_ptr + problem_id + tl.arange(0, 2), cache_modifier=".ca")
        start_idx, end_idx = group_indices.split()
        m = end_idx - start_idx
        
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_N)    
        num_tiles = tl.cast(num_m_tiles * num_n_tiles, tl.int32)
        tl.assume(num_tiles >= 0)
        end_tile_id = last_problem_end + num_tiles
        tiles_in_problem = tl.cdiv(end_tile_id - tile_id, NUM_PROGRAMS)
        tl.assume(tiles_in_problem >= 1)
        
        tl.assume(start_idx >= 0)
        tl.assume(end_idx >= start_idx)
        tl.assume(m >= 0)
        tl.assume(m <= NUM_TOKENS * TOPK)

        tile_id = m_grouped_gemm_inner_kernel(
            problem_id,
            tile_id,
            end_tile_id,
            start_idx,
            end_idx,
            last_problem_end,
            a_ptr,
            a_stride_1, a_stride_2,
            b_ptr,
            b_stride_1, b_stride_2, b_stride_3,
            out_ptr,
            out_stride_1, out_stride_2,
            preactivation_ptr,
            preactivation_stride_1, preactivation_stride_2,
            preact_for_grad_ptr,
            preact_for_grad_stride_1, preact_for_grad_stride_2,
            permute_indices_ptr,
            m,
            K,
            N,
            TOPK,
            GATHER_ROWS,
            SCATTER_ROWS,
            RETURN_PREACTIVATION,
            IS_A_TRANSPOSED,
            IS_B_TRANSPOSED,
            USE_A_TENSOR_DESCRIPTOR,
            USE_B_TENSOR_DESCRIPTOR,
            NUM_PROGRAMS,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            CACHE_GROUP_M,
            EPILOGUE,
            EPILOGUE_SPLIT,
            GRAD_ACT,
        )
    
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
    BLOCK_K = 64
    num_stages = 4
    if dtype == torch.float32:
        BLOCK_N //= 2
        num_stages -= 1
    default_config = triton.Config({
            "BLOCK_M": BLOCK_M, 
            "BLOCK_N": BLOCK_N, 
            "BLOCK_K": BLOCK_K, 
            "NUM_PROGRAMS": get_gpu_sm_count(),
            "CACHE_GROUP_M": 0,
            "EPILOGUE_SPLIT": 1,
            "DISALLOW_ACC_MULTI_BUFFER": False,
            "USE_A_TENSOR_DESCRIPTOR": False,
            "USE_B_TENSOR_DESCRIPTOR": False
        },
        num_warps=8, 
        num_stages=num_stages
    )
    return default_config

def _build_outputs(num_tokens, n, device, dtype, params: MGroupedGEMMParams):
    if params.gather or params.scatter:
        out_rows = num_tokens * params.topk
    else:
        out_rows = num_tokens
        
    out_cols = n
    if params.activation is not None and "glu" in params.activation:
        if "grad" in params.activation:
            out_cols *= 2
        else:
            out_cols //= 2
    out = torch.empty((out_rows, out_cols), device=device, dtype=dtype)
    
    if params.return_preactivation:
        preactivation = torch.empty((out_rows, n), device=device, dtype=dtype)
    else:
        preactivation = None
        
    return out, preactivation

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
    
    # if it's transposed, then rename n and k
    if params.is_b_transposed:
        k, n = n, k
    
    out, preactivation = _build_outputs(num_tokens, n, a.device, a.dtype, params)
    
    default_config = m_grouped_gemm_default_config(b.size(0), params, a.dtype)
    default_kwargs = default_config.all_kwargs()
    # torch.compile(fullgraph=True) does not supporting passing in num_ctas
    del default_kwargs["num_ctas"]
    
    block_k, block_n = default_kwargs["BLOCK_K"], default_kwargs["BLOCK_N"]
    use_b_tensor_descriptor = default_kwargs["USE_B_TENSOR_DESCRIPTOR"]
    del default_kwargs["USE_B_TENSOR_DESCRIPTOR"]
    # only use descriptors when all strides, except last, are 16-byte aligned
    use_b_tensor_descriptor = use_b_tensor_descriptor and all([(s * b.element_size()) % 16 == 0 for s in b.stride()[:-1]])
    if use_b_tensor_descriptor: 
        if params.is_b_transposed:
            block_size = [1, block_n, block_k]
        else:
            block_size = [1, block_k, block_n]
        b_desc = TensorDescriptor.from_tensor(b, block_size)

    use_a_tensor_descriptor = default_kwargs["USE_A_TENSOR_DESCRIPTOR"]
    del default_kwargs["USE_A_TENSOR_DESCRIPTOR"]
    use_a_tensor_descriptor = use_a_tensor_descriptor and all([(s * a.element_size()) % 16 == 0 for s in a.stride()[:-1]])
    if use_a_tensor_descriptor:
        # device TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)
    
    func = m_grouped_gemm_persistent_kernel
    if autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_m_grouped_gemm_persistent_kernel
        default_kwargs = {}
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_m_grouped_gemm_persistent_kernel
        default_kwargs = {}
    
    # TODO: torch.compile doesn't like passing in function    
    epilogue = params.activation

           
    grid = lambda META: (META["NUM_PROGRAMS"],)

    func[grid](
        a, 
        # torch.compile(fullgraph=True) does not supporting passing in tuples
        a.stride(0), a.stride(1),
        b_desc if use_b_tensor_descriptor else b,
        b.stride(0), b.stride(1), b.stride(2),
        out,
        out.stride(0), out.stride(1),
        preactivation if params.return_preactivation else None,
        preactivation.stride(0) if params.return_preactivation else None, 
        preactivation.stride(1) if params.return_preactivation else None,
        params.pre_act_for_grad,
        params.pre_act_for_grad.stride(0) if params.pre_act_for_grad is not None else None,
        params.pre_act_for_grad.stride(1) if params.pre_act_for_grad is not None else None,
        group_indices, 
        params.permute_indices, 
        NUM_TOKENS=num_tokens,
        NUM_EXPERTS=e, 
        K=k,
        N=n,
        TOPK=params.topk,
        GATHER_ROWS=params.gather,
        SCATTER_ROWS=params.scatter,
        RETURN_PREACTIVATION=params.return_preactivation,
        IS_A_TRANSPOSED=params.is_a_transposed,
        IS_B_TRANSPOSED=params.is_b_transposed,
        USE_A_TENSOR_DESCRIPTOR=use_a_tensor_descriptor,
        USE_B_TENSOR_DESCRIPTOR=use_b_tensor_descriptor,
        EPILOGUE=epilogue,
        GRAD_ACT=(params.pre_act_for_grad is not None) and "grad" in params.activation,
        **default_kwargs
    )
    
    return MGroupedGEMMOutput(output=out, preactivation=preactivation)