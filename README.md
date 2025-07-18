# MoE Explore

poking around some different MoE ideas and implementations.

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
in the home directory. So, this disables caching (caches to a temporary directory). 
Can also use `--cache-dir /path/to/.cache/uv` to set it somewhere else.

## Tests

```console
python -m pytest tests/
```
