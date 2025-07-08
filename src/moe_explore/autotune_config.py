from enum import Enum

class AutotuneMode(Enum):
    NONE = 0
    FAST = 1
    MAX = 2

DEFAULT_AUTOTUNE_MODE = AutotuneMode.FAST
