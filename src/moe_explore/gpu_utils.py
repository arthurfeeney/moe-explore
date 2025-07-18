import torch
from typing import Optional

def get_gpu_sm_count(device: Optional[torch.device] = None):
    # assuming this works for cuda and hip
    assert torch.cuda.device_count() > 0
    if device is None:
        device = torch.device(torch.cuda.current_device())
    return torch.cuda.get_device_properties(device).multi_processor_count
