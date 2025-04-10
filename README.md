# MoE Explore

poking around some different MoE ideas and implementations.

## Setup

```console
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/shawntan/scattermoe.git@main
pip install megablocks
```

## Structure

- `group_moe/` is a simple module for a group gemm MoE implementation. 
- `naive_moe/` is a basic moe implementation based on huggingface transformers.
- `profiling/` is scripts related to profiling and benchmarking different implementations.