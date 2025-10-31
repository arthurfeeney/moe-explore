import math
import torch
import torch.nn.functional as F
from moe_explore.functional.scale_and_reduce import scale_and_reduce
from moe_explore.triton_kernels.m_grouped_gemm import MGroupedGEMMParams
from moe_explore.params import MLPParams, TopkRouterParams, ErnieRouterParams
from moe_explore.functional.activation import activation

def torch_grouped_matmul_gather_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    group_indices: torch.Tensor,
    params: MGroupedGEMMParams,
):
    r"""
    This is a reference implementation of a grouped matmul, with an optional
    fused gather / scatter-reduce operation.
    """
    dtype = a.dtype
    a = a if not params.is_a_transposed else a.t().contiguous()
    b = b if not params.is_b_transposed else b.permute(0, 2, 1).contiguous()
    group_indices = group_indices
    gather_indices = params.permute_indices // params.topk if params.gather else None
    scatter_indices = params.permute_indices if params.scatter else None
    
    if params.gather:
        c_rows = a.size(0) * params.topk
    else:
        c_rows = a.size(0)    
    c = torch.zeros(c_rows, b.size(-1), device=a.device, dtype=dtype)
    
    for i in range(b.size(0)):
        if i == 0:
            glo = 0
            ghi = group_indices[i].item()
        else:
            glo = group_indices[i - 1].item()
            ghi = group_indices[i].item()
        #glo, ghi = group_indices[i].item(), group_indices[i + 1].item()
        if params.gather:
            index = gather_indices[glo:ghi].unsqueeze(-1).expand(-1, a.size(-1))
            a_gather = torch.gather(a, dim=0, index=index)
        else:
            a_gather = a[glo:ghi]

        prod = a_gather @ b[i]
        if params.scatter:
            c[scatter_indices[glo:ghi]] = prod
        else:
            c[glo:ghi] = prod
            
    if params.activation is not None:
        if params.pre_act_for_grad is not None:
            assert "grad" in params.activation
            c = activation(params.pre_act_for_grad, params.activation, grad_out=c)
        else:
            c = activation(c, params.activation, None)
             
    if params.scatter and params.scales is not None:
        c = scale_and_reduce(c, params.scales, params.num_tokens, params.topk, b.size(-1))
                
    return c.to(dtype)

def perfect_routing(num_tokens: int, num_experts: int, topk: int, device: torch.device, dtype: torch.dtype):
    r"""
    This generates a perfectly balanced routing, where each expert gets an equal number of tokens.
    This is useful for comparing performance with a batched GEMM.
    """
    assert num_tokens % num_experts == 0
    indices = torch.arange(0, num_tokens * topk, device=device) % num_experts
    topk_scores, _ = random_routing(num_tokens, num_experts, topk, device, dtype)
    return topk_scores, indices
    
def random_routing(num_tokens: int, num_experts: int, topk: int, device: torch.device, dtype: torch.dtype):
    r"""
    This generates a random routing with the routing logies selected from a normal distribution.
    This tends to generate a very balanced routing.
    """
    # Generate random scores. Applying softmax, even though it's nonsense numbers,
    # so the distribution looks like a normal MoE router output.
    scores = F.softmax(torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32), dim=-1)
    topk_logits, topk_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_logits.to(dtype), topk_indices

def random_skewed_routing(
    num_tokens: int,
    num_experts: int,
    topk: int, 
    num_skewed_experts: int,
    skew_factor: int, 
    device: torch.device, 
    dtype: torch.dtype
):
    r"""
    This generates a random routing based on a sort of simple skewed distribution.
    This makes it so that a subset of `num_skewed_experts` experts have a higher probability
    of being routed to than the rest. 
    The probabilities are determined by:
        A = [skew_factor if is_skewed(expert_id) else 1 for expert_id in range(num_experts)]
        Probs = A / sum(A).
    I.e., if we have four experts, and two are skewed, and skew_factor is 2, then the probabilities
    are [1, 1, 2, 2] / 6
    """
    skewed_experts_indices = torch.randperm(num_experts)[:num_skewed_experts]
    weights = torch.ones((num_tokens, num_experts), device=device, dtype=torch.float32)
    weights[:, skewed_experts_indices] *= skew_factor
    topk_indices = torch.multinomial(weights, num_samples=topk, replacement=False)
    topk_scores, _ = random_routing(num_tokens, num_experts, topk, device, dtype)
    return topk_scores, topk_indices

def random_groups(num_tokens, num_groups, device: torch.device):
    r"""
    Makes a `group_inidces` tensor like [0, r1, r2, ..., rN, num_tokens]. 
    The ri are random indices in [1, num_tokens - 1].
    """
    perm = torch.randperm(num_tokens - 1, device=device)
    group_indices = torch.empty(num_groups, device=device, dtype=torch.int32)
    group_indices[:-1] = perm[:num_groups - 1]
    group_indices[-1] = num_tokens
    group_indices = torch.sort(group_indices)[0]
    print(group_indices)
    return group_indices
    
    #indices = torch.arange(0, num_tokens - 1, device=device)
    #perm = torch.randperm(num_tokens - 1, device=device)
    #inner_group_indices, _ = torch.sort(indices[perm[:num_groups - 1]])
    #group_indices = torch.empty(num_groups, device=device, dtype=torch.int32)
    #group_indices[:-1] = inner_group_indices
    #group_indices[-1] = num_tokens
    #return group_indices

def uniform_weight_init(size, device, dtype):
    r"""
    This is based on pytorch's initialization for linear layers.
    1. For testing, we only use a uniform distribution. 
        The issue with testing against a normal distribution is that
        the output values can occassionally be quite large, which
        can make it difficult to set appropriate tolerances.
    2. MoE weights cannot use the default torch.nn.init functions because
        the fan_in / fan_out modes may not be "aware" that there is a batch
        of weights.  This somewhat arbitrary uses the last dim as the `fan` factor.
        (without explicitly noting that it's `fan_in` or `fan_out`--just tests, 
        so I don't think it really matters.)
    3. This is sort of arbitrarily chosen to be based on xavier uniform.
    """
    if len(size) == 1:
        weight = torch.empty(size, device=device, dtype=dtype)
        weight.uniform_(-0.02, 0.02)
        return weight
    
    if len(size) == 2:
        weight = torch.empty(size, device=device, dtype=dtype)
        torch.nn.init.xavier_uniform_(weight)
        return weight
    
    assert len(size) == 3
    weight = torch.empty(size, device=device, dtype=dtype)
    fan1, fan2 = size[1], size[2]
    std = math.sqrt(2 / (fan1 + fan2))
    a = math.sqrt(3) * std
    weight.uniform_(-a, a)
    return weight

def normal_weight_init(size, device, dtype):
    weight = torch.empty(size, device=device, dtype=dtype)
    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
    return weight

def random_mlp(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=uniform_weight_init,
):
    return MLPParams(
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype),
        activation
    )
    
def random_interleaved_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=uniform_weight_init,
):
    return MLPParams(
        dist((num_experts, hidden_dim, 2 * intermediate_dim), device=device, dtype=dtype),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype),
        activation
    )
    
def random_topk_router(
    num_experts,
    hidden_dim,
    topk,
    softmax_before_topk,
    normalize_routing,
    device, 
    dtype,
    dist=uniform_weight_init,
):
    return TopkRouterParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype),
        topk,
        softmax_before_topk=softmax_before_topk,
        normalize_routing=normalize_routing
    )
    
def random_ernie_router(
    num_experts,
    hidden_dim,
    topk,
    device,
    dtype,
    dist=uniform_weight_init,
):
    return ErnieRouterParams(
        dist((hidden_dim, num_experts), device=device, dtype=torch.float32),
        dist((num_experts,), device=device, dtype=torch.float32),
        topk,
    )
    
def assert_close(a, b, atol=None, rtol=None):
    # Tolerances depend on the matrix dimensions and the range of
    # values. A matrix with larger values and K-dimension will accumulate
    # more floating point errors... This tries to set tolerances
    # based on the dtype's epsilon, k-dimension, and max value.
    if atol is None:
        if a.dtype is not torch.float32:
            atol = 2e-2
        else:
            atol = 1e-4
    if rtol is None:    
        if a.dtype is torch.bfloat16:
            eps = 0.0078125
        elif a.dtype is torch.float16:
            eps = 0.0009765625
        elif a.dtype is torch.float32:
            eps = 1e-6
        else:
            raise ValueError(f"Invalid dtype: {a.dtype}")
        k = a.size(-1)
        rtol = math.log10(k) * eps
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)