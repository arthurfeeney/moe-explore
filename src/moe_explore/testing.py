import math
import torch
import torch.nn.functional as F
from moe_explore.triton_kernels.fused_moe import FusedMoeParams
from moe_explore.params import MLPParams, GLUParams

def torch_grouped_matmul_gather_scatter(
    a: torch.Tensor,
    params: FusedMoeParams,
):
    r"""
    This is a reference implementation of a grouped matmul, with an optional
    fused gather / scatter-reduce operation.
    """
    dtype = a.dtype
    a = a
    b = params.weight
    group_indices = params.group_indices
    gather_indices = params.permute_indices // params.topk if params.gather else None
    scatter_indices = params.permute_indices if params.scatter else None
    
    if params.gather:
        c_rows = a.size(0) * params.topk
    else:
        c_rows = a.size(0)    
    c = torch.zeros(c_rows, b.size(-1), device=a.device, dtype=dtype)
    
    for i in range(b.size(0)):
        glo, ghi = group_indices[i].item(), group_indices[i + 1].item()
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
            
    if params.scatter and params.scales is not None:
        c = (c.view(-1, params.topk, b.size(-1)) * params.scales[..., None]).sum(1)
            
    return c

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

def random_olmoe_routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: torch.device, 
    dtype: torch.dtype
):
    probabilities = torch.tensor([
        18.016000747680664,
        16.97599983215332,
        12.5,
        10.182000160217285,
        7.372000217437744,
        15.965999603271484,
        9.168000221252441,
        11.65999984741211,
        15.418000221252441,
        17.45800018310547,
        10.972000122070312,
        12.994000434875488,
        9.003999710083008,
        17.926000595092773,
        13.156000137329102,
        10.016000747680664,
        8.604000091552734,
        21.036001205444336,
        8.51200008392334,
        3.9600000381469727,
        2.615999937057495,
        6.663999557495117,
        8.369999885559082,
        16.756000518798828,
        18.18600082397461,
        2.8480000495910645,
        7.323999881744385,
        8.522000312805176,
        9.532000541687012,
        8.64799976348877,
        12.092000007629395,
        24.61400032043457,
        15.369998931884766,
        7.921999931335449,
        13.118000030517578,
        2.748000144958496,
        16.31399917602539,
        19.992000579833984,
        9.77400016784668,
        13.29800033569336,
        37.53199768066406,
        16.26799964904785,
        7.850000381469727,
        12.163999557495117,
        18.804000854492188,
        10.760000228881836,
        7.861999988555908,
        19.14000129699707,
        6.973999977111816,
        13.957999229431152,
        9.121999740600586,
        8.742000579833984,
        12.018000602722168,
        16.918001174926758,
        26.756000518798828,
        8.795999526977539,
        4.7260003089904785,
        16.59600067138672,
        7.269999980926514,
        11.458000183105469,
        9.90999984741211,
        6.0980000495910645,
        26.756000518798828,
        8.685999870300293,
    ], device=device, dtype=torch.float32) / 100.0
    topk_indices = torch.multinomial(probabilities.repeat(num_tokens, 1), num_samples=topk, replacement=False)
    topk_scores, _ = random_routing(num_tokens, num_experts, topk, device, dtype)
    return topk_scores, topk_indices


def random_groups(num_tokens, num_groups, device: torch.device):
    r"""
    Makes a `group_inidces` tensor like [0, r1, r2, ..., rN, num_tokens]. 
    The ri are random indices in [1, num_tokens - 1].
    """
    indices = torch.arange(1, num_tokens - 1, device=device)
    perm = torch.randperm(num_tokens - 2, device=device)
    inner_group_indices, _ = torch.sort(indices[perm[:num_groups - 1]])
    group_indices = torch.empty(num_groups + 1, device=device, dtype=torch.int32)
    group_indices[0] = 0
    group_indices[1:-1] = inner_group_indices
    group_indices[-1] = num_tokens
    return group_indices

def random_mlp(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=torch.randn,
):
    return MLPParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype) / math.sqrt(num_experts),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim),
        activation
    )

def random_glu(
    num_experts,
    hidden_dim,
    intermediate_dim,
    activation,
    device,
    dtype,
    dist=torch.randn,
):
    return GLUParams(
        dist((hidden_dim, num_experts), device=device, dtype=dtype) / math.sqrt(num_experts),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, hidden_dim, intermediate_dim), device=device, dtype=dtype) / math.sqrt(intermediate_dim),
        dist((num_experts, intermediate_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim),
        activation
    )