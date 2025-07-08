from enum import Enum

class AutotuneMode(Enum):
    NONE = 0
    FAST = 1
    MAX = 2

DEFAULT_AUTOTUNE_MODE = AutotuneMode.NONE

def get_matmul_autotune_configs(autotune_mode: AutotuneMode):
    if autotune_mode == AutotuneMode.NONE:
        return []
    if autotune_mode == AutotuneMode.FAST:
        return []
