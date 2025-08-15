from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh
from typing import Optional
from moe_explore.gpu_utils import get_gpu_sm_count
from .autotune_config import (
    AutotuneMode, 
    fast_autotune_configs, 
    max_autotune_configs
)

@dataclass
class GLUParams:
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    group_indices: torch.Tensor
    permute_indices: Optional[torch.Tensor]
    gather: bool
    num_tokens: int
    topk: int
    activation: str
    
@triton.jit
def silu(tensor):
    return tensor * tl.sigmoid(tensor)

@triton.jit
def gelu(tensor):
    # This is the approximation of gelu:
    #https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    pi: tl.constexpr = 3.14159265358979323846
    tensor_cubed = tensor * tensor * tensor
    return 0.5 * tensor * (1 + tanh(tl.sqrt(2 / pi) * (tensor + 0.044715 * tensor_cubed)))

@triton.jit
def m_grouped_glu_persistent_kernel(
    token_ptr,
    token_strides,
    gate_weight_ptr,
    up_weight_ptr,
    weight_strides, # Assuming gate_weight.stride == up_weight.stride()
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
    r"""
    This kernel loads 
        1. a [M, K] block from token_ptr.
        2. two [K, N] blocks from gate_weight and up_weight.
    The [M, K] block is expanded to be [2, M, K] and the weights
    are mushed into a [2, K, N] block.
    The result is a [2, M, N] block, which split into  `gate_acc` and `up_acc`.
    These are element-wise multipled and the [M, N] result is stored in `out_ptr`
    """
    tl.static_assert(K % BLOCK_K == 0, "K must be a multiple of BLOCK_K")
    tl.static_assert(N % BLOCK_N == 0, "N must be a multiple of BLOCK_N")
    tl.static_assert(ACTIVATION == "silu" or ACTIVATION == "gelu", "ACTIVATION must be either 'silu' or 'gelu'")

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

            gate_weight_ptrs = gate_weight_ptr + weight_problem_offset + weight_row_offsets + weight_col_offsets
            up_weight_ptrs = up_weight_ptr + weight_problem_offset + weight_row_offsets + weight_col_offsets

            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                tl.multiple_of(token_ptrs, [16, 16])
                tl.multiple_of(gate_weight_ptrs, [16, 16])
                tl.multiple_of(up_weight_ptrs, [16, 16])
                
                tokens_block = tl.load(token_ptrs, mask=token_mask[:, None], other=0.0)
                
                gate_weights_block = tl.load(gate_weight_ptrs)
                up_weights_block = tl.load(up_weight_ptrs)
                
                acc_gate = tl.dot(tokens_block, gate_weights_block, acc=acc_gate)
                acc_up = tl.dot(tokens_block, up_weights_block, acc=acc_up)
                
                # Batch matmul [2, BLOCK_M, BLOCK_N]
                #acc = tl.dot(tokens_block, weights, acc=acc) 
                token_ptrs += BLOCK_K * token_strides[1]
                gate_weight_ptrs += BLOCK_K * weight_strides[1]
                up_weight_ptrs += BLOCK_K * weight_strides[1]
            
            # Apply activation on float32 accumulator, because sigmoid and tanh need float32.
            if ACTIVATION == "silu":
                acc_gate = silu(acc_gate)
            elif ACTIVATION == "gelu":
                acc_gate = gelu(acc_gate)
            
            gated_acc = acc_gate * acc_up
            gated_acc = gated_acc.to(out_ptr.dtype.element_ty)
            
            out_row_offsets = start_idx + tile_m_offsets
            out_offsets = out_row_offsets[:, None] * out_strides[0] + tile_n_offsets * out_strides[1]
            out_ptrs = out_ptr + out_offsets
            tl.store(out_ptrs, gated_acc, mask=token_mask[:, None])

            tile_id += NUM_PROGRAMS
        last_problem_end += num_tiles 

_fast_autotune_m_grouped_glu_persistent_kernel = triton.autotune(
    configs=fast_autotune_configs(persistent=True),
    key=['E', 'N', 'K'],
    reset_to_zero=['out_ptr']
)(m_grouped_glu_persistent_kernel)

_max_autotune_m_grouped_glu_persistent_kernel = triton.autotune(
    configs=max_autotune_configs(persistent=True),
    key=['E', 'N', 'K'],
    reset_to_zero=['out_ptr']
)(m_grouped_glu_persistent_kernel)

def glu(
    token: torch.Tensor,
    params: GLUParams,
    autotune_mode: Optional[AutotuneMode] = None
):
    assert token.dim() == 2
    assert params.gate_weight.dim() == 3
    assert params.up_weight.dim() == 3
    assert params.gate_weight.size() == params.up_weight.size()
    assert token.size(1) == params.gate_weight.size(1)
    assert autotune_mode is None or autotune_mode in AutotuneMode
    assert params.activation in ("silu", "gelu")

    if params.gather:
        assert params.permute_indices is not None

    num_tokens = params.num_tokens
    e, k, n = params.gate_weight.size()

    default_kwargs = {}
    if autotune_mode is None or autotune_mode == AutotuneMode.NONE:
        default_kwargs = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_PROGRAMS": get_gpu_sm_count(), "num_warps": 4, "num_stages": 4}
        func = m_grouped_glu_persistent_kernel
    elif autotune_mode == AutotuneMode.FAST:
        func = _fast_autotune_m_grouped_glu_persistent_kernel
    elif autotune_mode == AutotuneMode.MAX:
        func = _max_autotune_m_grouped_glu_persistent_kernel

    if params.gather:
        c_rows = num_tokens * params.topk
    else:
        c_rows = num_tokens

    out = torch.empty((c_rows, n), device=token.device, dtype=token.dtype)
    
    grid = lambda META: (META["NUM_PROGRAMS"],)

    func[grid](
        token, 
        token.stride(),
        params.gate_weight,
        params.up_weight,
        params.gate_weight.stride(),
        out,
        out.stride(),
        params.group_indices, 
        params.permute_indices, 
        ACTIVATION=params.activation,
        ACC_DTYPE=tl.float32,
        E=e, 
        K=k,
        N=n,
        TOPK=params.topk,
        GATHER_ROWS=params.gather,
        **default_kwargs
    )

    return out