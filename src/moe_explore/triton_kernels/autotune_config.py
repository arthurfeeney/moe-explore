from dataclasses import dataclass, astuple
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
    params = [block_m, block_n, block_k, num_warps, num_stages]
    if persistent:
        params.append(AutotuneParam("NUM_PROGRAMS", [get_gpu_sm_count()]))
    return generate_configs(params)

def max_autotune_configs(persistent: bool):
    block_sizes = [32, 64, 128, 256]
    block_m = AutotuneParam("BLOCK_M", block_sizes)
    block_n = AutotuneParam("BLOCK_N", block_sizes)
    block_k = AutotuneParam("BLOCK_K", [32, 64, 128])
    group_sizes = AutotuneParam("GROUP_M", [0, 4, 6, 8])
    num_warps = AutotuneParam("num_warps", [4, 8, 16])
    num_stages = AutotuneParam("num_stages", [3, 4, 5])
    params = [block_m, block_n, block_k, num_warps, num_stages, group_sizes]
    if persistent:
        sm_count = get_gpu_sm_count()
        num_programs = [sm_count]
        params.append(AutotuneParam("NUM_PROGRAMS", num_programs))
    return generate_configs(params)