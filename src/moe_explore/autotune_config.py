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
        configs.append(triton.Config(dict_params))
    return configs

def fast_autotune_configs(persistent: bool):
    block_sizes = [64, 128]
    block_m = AutotuneParam("BLOCK_M", block_sizes)
    block_n = AutotuneParam("BLOCK_N", block_sizes)
    block_k = AutotuneParam("BLOCK_K", block_sizes)
    params = [block_m, block_n, block_k]
    if persistent:
        params.append(AutotuneParam("NUM_PROGRAMS", [get_gpu_sm_count()]))
    return generate_configs(params)
    
def max_autotune_configs(persistent: bool):
    block_sizes = [64, 128, 256]
    block_m = AutotuneParam("BLOCK_M", block_sizes)
    block_n = AutotuneParam("BLOCK_N", block_sizes)
    block_k = AutotuneParam("BLOCK_K", block_sizes)
    num_warps = AutotuneParam("num_warps", [4, 8]) 
    num_stages = AutotuneParam("num_stages", [3, 4])
    params = [block_m, block_n, block_k, num_warps, num_stages]
    if persistent:
        sm_count = get_gpu_sm_count()
        num_programs = [sm_count // 2, sm_count, sm_count * 2]
        params.append(AutotuneParam("NUM_PROGRAMS", num_programs))
    return generate_configs(params)
