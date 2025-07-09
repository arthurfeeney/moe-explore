import torch

def get_gpu_sm_count(device: torch.device = None):
    # assuming this works for cuda and hip
    if device is None:
        device = torch.get_default_device()
    return torch.cuda.get_device_properties(device).multi_processor_count
