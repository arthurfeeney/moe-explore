import torch
import triton
import triton.language as tl
from typing import List
from expert_permute import GroupedTokens

DEVICE = 'cuda'

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

configs = [
    # M, N, K, warps, stages
    (128, 128, 128, 4, 4),
    (128, 128, 64, 4, 4),
    (128, 128, 32, 4, 4),
    (256, 256, 64, 4, 4),
    (64, 128, 128, 4, 4),
    (128, 128, 128, 8, 4),
    (256, 256, 64, 8, 4),
    (64, 64, 128, 4, 4),
    (64, 64, 256, 4, 4),
    (128, 64, 256, 4, 4)
]

configs = [
    triton.Config({
        'BLOCK_SIZE_M': c[0],
        'BLOCK_SIZE_N': c[1],
        'BLOCK_SIZE_K': c[2],
        'num_warps': c[3],
        'num_stages': c[4],
        'NUM_SM': num_sms()
    }) for c in configs
]


@triton.autotune(
    configs=configs,
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            
            m_mask = offs_am < gm
            n_mask = offs_bn < gn
            
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
            
                k_mask = kk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < k
                                
                a = tl.load(a_ptrs, mask=m_mask[:, None] and k_mask)
                b = tl.load(b_ptrs, mask=k_mask[:, None] and n_mask)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            tl.store(c_ptrs, c, mask=m_mask[:, None] and n_mask)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

def get_a_addrs(group_A: GroupedTokens, group_size: int):
    a_addrs = []
    a_shapes = []
    for i in range(group_size):
        end_idx = group_A.tokens_per_expert_range_cpu[i]
        start_idx = 0 if i == 0 else group_A.tokens_per_expert_range_cpu[i - 1]
        a_addrs.append(group_A.tokens[start_idx:-1].data_ptr())
        M = end_idx - start_idx
        K = group_A.tokens.size(-1)
        a_shapes.append((M, K))
    return a_addrs, a_shapes
        
def get_b_addrs(group_B: torch.Tensor, group_size: int):
    b_addrs = []
    b_shapes = []
    for i in range(group_size):
        b_addrs.append(group_B[i].data_ptr())
        K = group_B[i].size(0)
        N = group_B[i].size(1)
        b_shapes.append((K, N))
    return b_addrs, b_shapes

def group_gemm_fn(group_A: GroupedTokens, group_B: torch.Tensor):
    group_size = group_B.size(0)
    
    A_addrs, A_shapes = get_a_addrs(group_A, group_size)
    B_addrs, B_shapes = get_b_addrs(group_B, group_size)
    
    C_addrs = []
    g_sizes = []
    g_lds = []
    
    assert (sum([s[0] for s in A_shapes]) == group_A.tokens.size(0))
    
    c = torch.ones((group_A.tokens.size(0), group_B.size(-1)), device=DEVICE, dtype=group_A.tokens.dtype)
    
    c_offset = 0
    for i in range(group_size):
        assert A_shapes[i][1] == B_shapes[i][0]
        M, K = A_shapes[i]
        K, N = B_shapes[i]
        
        C_addrs.append(c[c_offset:].data_ptr())
        g_sizes += [M, N, K]
        g_lds += [K, N, c.stride(0)]
        c_offset += M
                
    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    )
    
    print(group_A.tokens)
    print(c)
    
    assert torch.allclose(group_A.tokens, c)
        
    return GroupedTokens(
        tokens=c,
        tokens_per_expert_range=group_A.tokens_per_expert_range,
        tokens_per_expert_range_cpu=group_A.tokens_per_expert_range_cpu,
        indices=group_A.indices,
        token_gather_indices=group_A.token_gather_indices,
    )