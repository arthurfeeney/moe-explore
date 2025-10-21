import triton
import triton.language as tl

@triton.jit
def get_tile_id_in_group(
    tile_id_in_gemm,
    m,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CACHE_GROUP_M: tl.constexpr,
):
    # Triton does not like computing // or % when num_n_tiles is passed in as a paramter.
    # MLIR inliner fails. So, this recomputes it.
    if CACHE_GROUP_M == 0:
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        return tile_id_in_gemm // num_n_tiles * BLOCK_M, tile_id_in_gemm % num_n_tiles * BLOCK_N
    else:
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        group_m = CACHE_GROUP_M
        num_tiles_in_group = group_m * num_n_tiles
        group_id = tile_id_in_gemm // num_tiles_in_group
        first_id_m = group_id * group_m
        group_size_m = min(num_m_tiles - first_id_m, group_m)
        tile_m_idx = (first_id_m + ((tile_id_in_gemm % num_tiles_in_group) % group_size_m)) * BLOCK_M
        tile_n_idx = ((tile_id_in_gemm % num_tiles_in_group) // group_size_m) * BLOCK_N
        return tile_m_idx, tile_n_idx

@triton.jit
def tile_offsets_in_group(
    tile_m_idx,
    tile_n_idx,
    m,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    tile_m_offsets = tile_m_idx + tl.arange(0, BLOCK_M)
    tile_n_offsets = tile_n_idx + tl.arange(0, BLOCK_N)
    tile_m_offsets = tl.max_contiguous(tl.multiple_of(tile_m_offsets % m, BLOCK_M), BLOCK_M)
    tile_n_offsets = tl.max_contiguous(tl.multiple_of(tile_n_offsets % N, BLOCK_N), BLOCK_N)
    return tile_m_offsets, tile_n_offsets