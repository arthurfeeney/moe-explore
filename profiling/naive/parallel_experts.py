import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelExperts(nn.Module):
    def __init__(self, num_experts, embed_dim, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.topk = 4
        self.gate = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_dim = x.size(2)
        x = x.view(-1, self.hidden_dim)
        routing_logits = self.gate(x)
        routing_weights = torch.softmax(routing_logits.float(), dim=1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        final = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        
        for idx in range(self.num_experts):
            expert = self.experts[idx]
            idx, top_x = torch.where(expert_mask[idx])
            cur_state = x[None, top_x].reshape(-1, self.hidden_dim)
            
            cur_hidden_states = expert(cur_state) * routing_weights[top_x, idx, None]
            
            final.index_add_(0, top_x, cur_hidden_states.to(x.dtype))
            
        final = final.reshape(batch_size, seq_len, hidden_dim)
        return final, routing_logits

pe = ParallelExperts(8, 128, 128)
print(pe)

data = torch.randn(8, 16, 128)

f = pe(data)

print(f)