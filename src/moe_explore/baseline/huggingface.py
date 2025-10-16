import torch

try:
    # The olmoe, qwen, etc SparseMoeBlocks are basically copy-pastes of mixtral's,
    # so just use the OlmoeSparseMoeBLock for the sake of keeping it simple.
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    HAVE_HUGGINGFACE = True
except ImportError:
    HAVE_HUGGINGFACE = False

def make_huggingface_moe(
    input_dim: int,
    hidden_dim: int,
    num_experts: int,
    topk: int,
    activation: str,
    dtype: torch.dtype
):
    config = OlmoeConfig(
        hidden_size=input_dim,
        intermediate_size=hidden_dim,
        num_experts=num_experts,
        num_experts_per_tok=topk
    )
    moe = OlmoeSparseMoeBlock(config)
    return moe

def set_huggingface_moe_weights(moe: OlmoeSparseMoeBlock, router_weight, weight1, weight2):
    moe.gate.weight.data[:] = router_weight.t()
    for i in range(moe.num_experts):
        moe.experts[i].gate_proj.weight.data[:] = weight1[i, :, 0::2].t()
        moe.experts[i].up_proj.weight.data[:] = weight1[i, :, 1::2].t()
        moe.experts[i].down_proj.weight.data[:] = weight2[i].t()
    return moe