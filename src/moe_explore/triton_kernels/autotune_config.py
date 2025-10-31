from dataclasses import dataclass
from enum import StrEnum
import itertools
import triton
from typing import List

from moe_explore.gpu_utils import get_gpu_sm_count

class AutotuneMode(StrEnum):
    NONE = "none"
    FAST = "fast"
    MAX = "max"

@dataclass
class AutotuneParam:
    key: str
    value: List[int]

def generate_configs(params: List[AutotuneParam]):
    configs = []
    params = [p for p in params if p is not None]
    keys = [p.key for p in params]
    values = [p.value for p in params]
    for combo in itertools.product(*values):
        dict_params = dict(zip(keys, combo))
        num_warps = dict_params.get("num_warps", 4)
        num_stages = dict_params.get("num_stages", 3)
        if "num_warps" in dict_params:
            del dict_params["num_warps"]
        if "num_stages" in dict_params:
            del dict_params["num_stages"]
        configs.append(triton.Config(dict_params, num_warps=num_warps, num_stages=num_stages))
    return configs

def fast_autotune_configs(persistent: bool):
    block_m = AutotuneParam("BLOCK_M", [128])
    block_n = AutotuneParam("BLOCK_N", [128, 256])
    block_k = AutotuneParam("BLOCK_K", [32, 64])
    num_warps = AutotuneParam("num_warps", [4, 8])
    num_stages = AutotuneParam("num_stages", [4])
    epilogue_split = AutotuneParam("EPILOGUE_SPLIT", [1, 2])
    group_sizes = AutotuneParam("CACHE_GROUP_M", [0, 4, 6])
    params = [block_m, block_n, block_k, num_warps, num_stages, epilogue_split, group_sizes]
    if persistent:
        params.append(AutotuneParam("NUM_PROGRAMS", [get_gpu_sm_count()]))
    return generate_configs(params)

def max_autotune_configs(persistent: bool, k_loop_stages: bool = False):
    block_sizes = [64, 128, 256]
    block_m = AutotuneParam("BLOCK_M", block_sizes)
    block_n = AutotuneParam("BLOCK_N", block_sizes)
    block_k = AutotuneParam("BLOCK_K", [32, 64])
    group_sizes = AutotuneParam("CACHE_GROUP_M", [0, 4, 6, 8])
    num_warps = AutotuneParam("num_warps", [4, 8])
    if k_loop_stages:
        num_stages = AutotuneParam("K_LOOP_STAGES", [3, 4, 5])
    else:
        num_stages = AutotuneParam("num_stages", [3, 4, 5])
    epilogue_split = AutotuneParam("EPILOGUE_SPLIT", [1, 2])
    params = [block_m, block_n, block_k, num_warps, num_stages, group_sizes, epilogue_split]
    if persistent:
        sm_count = get_gpu_sm_count()
        num_programs = [sm_count]
        params.append(AutotuneParam("NUM_PROGRAMS", num_programs))
    return generate_configs(params)