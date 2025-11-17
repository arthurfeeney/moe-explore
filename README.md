# MoE Explore

This project provides GPU kernels that can be used for fine-tuning mixture-of-experts modules.
The main kernels are an M-Grouped GEMM and K-Grouped GEMM, which support different levels of
operator fusion for token routing.  
It also provides a simple interface for running an MoE with a top-k router:

```python
from moe_explore.moe import TopkMoE
moe = TopkMoE(
    num_experts=128,
    hidden_dim=1024,
    intermediate_dim=512,
    activation="swiglu"
)
output = moe(tokens)
output.sum().backward() # backward pass is also supported
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv)

```console
uv venv
source .venv/bin/activate
uv sync --no-cache
uv pip install -e .
```

Once installed, you can run the tests:

```console
python -m pytest tests/
```