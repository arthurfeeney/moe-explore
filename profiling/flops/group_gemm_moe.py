r"""
This implements MoE via rearranging everything and following with a group gemm.
"""

import torch
from torch import nn

class GroupGEMMMoE(nn.Module):
    def __init__(
        self,
        num_experts,
        input_size,
        output_size
    ):
        self.experts = torch.nn.Parameter(torch.randn(
            (num_experts, input_size, output_size)
        ))
        
    def forward(self, x, topk_indices):
        r"""
        Args:
            x: [b, s, d], input data
            topk_indices: [b, s, topk], indices of experts to route to.
            I.e., topk_indices[i, j] is the indices of the experts that
            x[i, j] should be routed to.
        """
        pass

b = 1
s = 4
d = 16
experts = 8
k = 2

x = torch.randn((b, s, d), device='cuda', dtype=torch.float16)
topk_indices = torch.randint(
    low=0, high=experts, size=(b, s, k), device='cuda').to(torch.int64)

for e in range(experts):
    x[topk_indices == e]