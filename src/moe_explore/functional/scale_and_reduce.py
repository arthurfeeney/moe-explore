import torch

# There isn't a great place for this. It is needed for both the GLU and MLP.
@torch.compile
def scale_and_reduce(out, scales, num_tokens, topk, n):
    return (out.view(num_tokens, topk, n) * scales[..., None]).sum(1)
