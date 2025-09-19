import torch

@torch.compile
def scale_and_reduce(input, scales, num_tokens, topk, n):
    return (input.view(num_tokens, topk, -1) * scales[..., None]).sum(1)