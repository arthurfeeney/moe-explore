import torch
from typing import Optional

def get_gpu_sm_count(device: Optional[torch.device] = None):
    # assuming this works for cuda and hip
    if device is None:
        device = torch.device(torch.cuda.current_device())
    return torch.cuda.get_device_properties(device).multi_processor_count

def get_gpu_sm_version(device: Optional[torch.device] = None):
    if device is None:
        device = torch.device(torch.cuda.current_device())
    prop = torch.cuda.get_device_properties(device)
    return prop.major * 10 + prop.minor