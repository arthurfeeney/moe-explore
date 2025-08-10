# MoE Explore

poking around some different MoE ideas and implementations.
This is currently focused on single-gpu kernels.

## Setup

This project uses [uv](https://github.com/astral-sh/uv)

```console
uv venv
source .venv/bin/activate
uv sync --no-cache
uv pip install -e .
```

uv caches downloads very aggressively. The cache can easily be
dozens of gigabytes. By default, uv caches to the home directory, `~`.
A lot of university clusters have a limit for the amount of data that can be
in the home directory. So, this disables caching (it instead caches to a temporary directory). 
You can also use `--cache-dir /path/to/.cache/uv` to set it somewhere else.

On an NFS, python imports are insanely slow, especially for larger
projects, like huggingface. Can just install dependencies on the compute node.
This is slightly annoying because this has to be done every time you get a node...
BUT, the speedup should be immediately obvious and very large.

```console
uv venv $TMPDIR/moe-explore
source $TMPDIR/moe-explore/bin/activate
uv sync --no-cache --active --all-extras
pip install -e .
```

## Performance Comparisons

Stuff not on pypi, for now clone and install manually

- [ScatterMoE](https://github.com/shawntan/scattermoe)
- [Unsloth MoE](https://github.com/unslothai/unsloth/tree/main/unsloth/kernels/moe)
- Triton Kernels

## Tests

```console
python -m pytest tests/
```
