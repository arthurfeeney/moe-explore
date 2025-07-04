import torch

def get_gpu_sm_count(device):
    # assuming this works for cuda and hip
    return torch.cuda.get_device_properties(device).multi_processor_count
