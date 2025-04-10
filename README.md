# MoE Explore

Just poking around some different MoE ideas and implementations.

## Setup

```console
pip install torch
```

## Structure

- `profiling/` is scripts related to profiling and benchmarking different implementations.

## Other Implementations

- [ScatterMoE](https://github.com/shawntan/scattermoe)
- [MegaBlocks](https://github.com/databricks/megablocks)

HuggingFace transformers also has a lot of differnet MoE implementations, but they
seem to all be simple loops over experts.