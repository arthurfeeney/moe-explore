import torch
from moe_explore.functional.topk_moe import topk_moe
from moe_explore.params import MOEParams, MLPParams, TopkRouterParams

class TopkMoE(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        topk: int,
        activation: str,
        return_topk_logits: bool = False
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.topk = topk
        self.activation = activation
        self.return_topk_logits = return_topk_logits

        self.router_weight = torch.nn.Parameter(torch.empty(hidden_dim, num_experts))
        weight1_dim = intermediate_dim * 2 if "glu" in activation else intermediate_dim
        self.weight1 = torch.nn.Parameter(torch.empty(num_experts, hidden_dim, weight1_dim))
        self.weight2 = torch.nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim))
        
        self.reset_params()

    def reset_params(self):
       with torch.no_grad():
           self.router_weight.normal_(mean=0.0, std=0.023)
           self.weight1.normal_(mean=0.0, std=0.023)
           self.weight2.normal_(mean=0.0, std=0.023)
       
    def forward(self, input: torch.tensor):
        params = MOEParams(
            TopkRouterParams(
                router_weight=self.router_weight,
                topk=self.topk,
                softmax_before_topk=True,
                normalize_routing=False
            ),
            MLPParams(
                weight1=self.weight1,
                weight2=self.weight2,
                activation=self.activation
            ),
            num_experts=self.num_experts,
            topk=self.topk
        )
        
        flat_input = input.view(-1, input.size(-1))
        moe_output = topk_moe(flat_input, params, return_router_logits=self.return_topk_logits)
        if self.return_topk_logits:
            output, topk_logits = moe_output
            return output.view(input.size()), topk_logits
        return moe_output