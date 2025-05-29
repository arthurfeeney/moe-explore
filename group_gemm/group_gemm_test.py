import torch
from torch import nn
from expert_permute import expert_input_permute, expert_output_permute
from group_gemm_triton import group_gemm_fn

def make_identity_experts(num_experts, input_dim, output_dim):
    data = torch.zeros(num_experts, input_dim, output_dim)
    data[:, range(input_dim), range(output_dim)] = 1
    return data

class MoERouter(nn.Module):
    def __init__(self, num_experts, input_dim, topk):
        super().__init__()
        self.weights = nn.Linear(input_dim, num_experts, bias=False)
        #self.identity_init()
        self.topk = topk

    def identity_init(self):
        with torch.no_grad():
            self.weights.weight.copy_(torch.eye(*self.weights.weight.size()))
        
    def forward(self, tokens):
        scores = self.weights(tokens).softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.topk, dim=-1, sorted=False)
        return topk_scores, topk_indices
    
class ExpertsGroupGEMM(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.weights1 = nn.Parameter(make_identity_experts(num_experts, input_dim, hidden_dim))        
        self.weights2 = nn.Parameter(make_identity_experts(num_experts, hidden_dim, output_dim))
        self.act = torch.nn.functional.gelu

    def forward(self, group_token):
        group_weights1 = [w for w in self.weights1]
        group_weights2 = [w for w in self.weights2]
                
        group_token = group_gemm_fn(group_token, group_weights1)
        group_token = [self.act(t) for t in group_token] # TODO: fuse into group gemm.
        group_token = group_gemm_fn(group_token, group_weights2)
        
        return group_token

class MoEGroupGEMM(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, topk):
        super().__init__()
        self.topk = topk
        self.experts = ExpertsGroupGEMM(num_experts, input_dim, input_dim, output_dim)
        self.router = MoERouter(num_experts, input_dim, self.topk)
        
    def forward(self, tokens):
        expert_scores, expert_indices = self.router(tokens)
        group_token, _ = expert_input_permute(tokens, expert_indices, self.topk)
        group_token = self.experts(group_token)
        output = expert_output_permute(group_token, expert_indices, expert_scores, self.topk, tokens.shape)
        return output
    
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.weights1 = nn.Parameter(torch.eye(input_dim, hidden_dim))        
        self.weights2 = nn.Parameter(torch.eye(hidden_dim, output_dim))
        self.act = torch.nn.functional.gelu
        
    def forward(self, tokens):
        tokens = tokens @ self.weights1
        tokens = self.act(tokens)
        tokens = tokens @ self.weights2
        return tokens

class MoENaive(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, topk):
        super().__init__()
        self.topk = topk
        self.router = MoERouter(num_experts, input_dim, self.topk)
        self.experts = nn.ModuleList([Expert(input_dim, input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, tokens):
        expert_scores, expert_indices = self.router(tokens)
        _, hidden_dim = tokens.shape
        orig_shape = tokens.shape
        x_flat = tokens.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(x_flat,
                                            flat_expert_indices,
                                            flat_expert_weights)

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output
        
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        token_idxs = idxs // self.topk
        print('naive token idxs', token_idxs)
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )

        return expert_cache

device = 'cuda'
dtype = torch.float16

num_tokens = 4
tokens = torch.randn(num_tokens, 128).to(dtype).to(device)

# apply 1 expert, identity weights, but with acitvation function.
moe = MoEGroupGEMM(1, 128, 128, 1).to(dtype).to(device)
output_group_gemm = moe(tokens)
moe = MoENaive(1, 128, 128, 1).to(dtype).to(device)
output_naive = moe(tokens)
print(output_group_gemm.mean())
print(output_naive.mean())
assert torch.allclose(output_group_gemm, output_naive)

# apply 4 expert, top 2, identity weights.
# set seeds so the routers are the same.
# TODO: just pass the same router as argument into both modules
torch.manual_seed(0)
moe = MoEGroupGEMM(4, 128, 128, 2).to(dtype).to(device)
output_group_gemm = moe(tokens)

torch.manual_seed(0)
moe = MoENaive(4, 128, 128, 2).to(dtype).to(device)
output_naive = moe(tokens)

print(output_group_gemm.mean())
print(output_naive.mean())

assert torch.allclose(output_group_gemm, output_naive)