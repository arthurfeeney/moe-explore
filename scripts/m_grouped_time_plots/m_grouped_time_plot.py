import torch
import pandas
import seaborn
import matplotlib.pyplot as plt
import pathlib

seaborn.set_theme(
    font_scale=1.3, 
    style="white",
    palette=None,
    rc={"figure.constrained_layout.use": True}
)

# TODO: The CSV files should all be moved into this directory.

#
# 1. Make plots for qwen3
#

def rename_columns(df):
    df = df.rename(columns={
        "Torch Grouped MM": "torch._grouped_mm",
        "Grouped-only": "m-grouped-gemm",
        "Grouped+Gather": "m-grouped-gemm + gather",
        "Grouped+Scatter": "m-grouped-gemm + scatter",
    })
    return df

# qwen3 on A30
qwen_a30_dir = pathlib.Path("microbenchmarks/m_grouped_gemm/NVIDIA A30/1761666825.164804/")
fix, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_a30_perfect_routing = rename_columns(pandas.read_csv(qwen_a30_dir / "Qwen3-30B-A3B-style GEMM2, perfect routing.csv"))
qwen3_a30_balanced_routing = rename_columns(pandas.read_csv(qwen_a30_dir / "Qwen3-30B-A3B-style GEMM2, balanced routing.csv"))
qwen3_a30_skewed_routing = rename_columns(pandas.read_csv(qwen_a30_dir / "Qwen3-30B-A3B-style GEMM2, skewed routing.csv"))
qwen3_a30_perfect_routing.plot(x="num_tokens", ax=ax[0], legend=False)
qwen3_a30_balanced_routing.plot(x="num_tokens", ax=ax[1], legend=False)
qwen3_a30_skewed_routing.plot(x="num_tokens", ax=ax[2], legend=False)

ax[0].set_title("Perfect Routing", fontsize=18)
ax[1].set_title("Balanced Routing", fontsize=18)
ax[2].set_title("Skewed Routing", fontsize=18)
ax[0].set_ylabel("TFLOP/s")
ax[0].set_xlabel("Number of Tokens")
ax[1].set_xlabel("Number of Tokens")
ax[2].set_xlabel("Number of Tokens")

plt.legend()
plt.savefig("m_grouped_time_plot_qwen3_a30.pdf")
plt.close()

# qwen3 on A100
qwen_a100_dir = pathlib.Path("microbenchmarks/m_grouped_gemm/NVIDIA A100 80GB PCIe/1761653889.8838954/")
fix, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_a100_perfect_routing = rename_columns(pandas.read_csv(qwen_a100_dir / "Qwen3-30B-A3B-style GEMM2, perfect routing.csv"))
qwen3_a100_balanced_routing = rename_columns(pandas.read_csv(qwen_a100_dir / "Qwen3-30B-A3B-style GEMM2, balanced routing.csv"))
qwen3_a100_skewed_routing = rename_columns(pandas.read_csv(qwen_a100_dir / "Qwen3-30B-A3B-style GEMM2, skewed routing.csv"))
qwen3_a100_perfect_routing.plot(x="num_tokens", ax=ax[0], legend=False)
qwen3_a100_balanced_routing.plot(x="num_tokens", ax=ax[1], legend=False)
qwen3_a100_skewed_routing.plot(x="num_tokens", ax=ax[2], legend=False)

ax[0].set_ylabel("TFLOP/s")
ax[0].set_xlabel("Number of Tokens")
ax[1].set_xlabel("Number of Tokens")
ax[2].set_xlabel("Number of Tokens")

plt.legend()
plt.savefig("m_grouped_time_plot_qwen3_a100.pdf")
plt.close()

# qwen3 on H100 NVL
qwen_h100_dir = pathlib.Path("moe-explore-results-h100/microbenchmarks/m_grouped_gemm/NVIDIA H100 NVL/1761700440.2513993/")
fix, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_h100_perfect_routing = rename_columns(pandas.read_csv(qwen_h100_dir / "Qwen3-30B-A3B-style GEMM2, perfect routing.csv"))
qwen3_h100_balanced_routing = rename_columns(pandas.read_csv(qwen_h100_dir / "Qwen3-30B-A3B-style GEMM2, balanced routing.csv"))
qwen3_h100_skewed_routing = rename_columns(pandas.read_csv(qwen_h100_dir / "Qwen3-30B-A3B-style GEMM2, skewed routing.csv"))    
qwen3_h100_perfect_routing.plot(x="num_tokens", ax=ax[0], legend=False)
qwen3_h100_balanced_routing.plot(x="num_tokens", ax=ax[1], legend=False)
qwen3_h100_skewed_routing.plot(x="num_tokens", ax=ax[2], legend=False)

ax[0].set_ylabel("TFLOP/s")
ax[0].set_xlabel("Number of Tokens")
ax[1].set_xlabel("Number of Tokens")
ax[2].set_xlabel("Number of Tokens")

plt.legend()
plt.savefig("m_grouped_time_plot_qwen3_h100.pdf")
plt.close()

#
# 1. Make plots for olmoe
#

def rename_columns(df):
    df = df.rename(columns={
        "Torch Grouped MM": "torch._grouped_mm",
        "Grouped-only": "m-grouped-gemm",
        "Grouped+Gather": "m-grouped-gemm + gather",
        "Grouped+Scatter": "m-grouped-gemm + scatter",
    })
    return df

# olmoe on A30
olmoe_a30_dir = pathlib.Path("microbenchmarks/m_grouped_gemm/NVIDIA A30/1761666825.164804/")
fix, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
olmoe_a30_perfect_routing = rename_columns(pandas.read_csv(olmoe_a30_dir / "OLMoE-1B-7B-style GEMM2, perfect routing.csv"))
olmoe_a30_balanced_routing = rename_columns(pandas.read_csv(olmoe_a30_dir / "OLMoE-1B-7B-style GEMM2, balanced routing.csv"))
olmoe_a30_skewed_routing = rename_columns(pandas.read_csv(olmoe_a30_dir / "OLMoE-1B-7B-style GEMM2, skewed routing.csv"))
olmoe_a30_perfect_routing.plot(x="num_tokens", ax=ax[0], legend=False)
olmoe_a30_balanced_routing.plot(x="num_tokens", ax=ax[1], legend=False)
olmoe_a30_skewed_routing.plot(x="num_tokens", ax=ax[2], legend=False)

ax[0].set_title("Perfect Routing", fontsize=18)
ax[1].set_title("Balanced Routing", fontsize=18)
ax[2].set_title("Skewed Routing", fontsize=18)
ax[0].set_ylabel("TFLOP/s")
ax[0].set_xlabel("Number of Tokens")
ax[1].set_xlabel("Number of Tokens")
ax[2].set_xlabel("Number of Tokens")

plt.legend()
plt.savefig("m_grouped_time_plot_olmoe_a30.pdf")
plt.close()

# olmoe on A100
olmoe_a100_dir = pathlib.Path("microbenchmarks/m_grouped_gemm/NVIDIA A100 80GB PCIe/1761653889.8838954/")
fix, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
olmoe_a100_perfect_routing = rename_columns(pandas.read_csv(olmoe_a100_dir / "OLMoE-1B-7B-style GEMM2, perfect routing.csv"))
olmoe_a100_balanced_routing = rename_columns(pandas.read_csv(olmoe_a100_dir / "OLMoE-1B-7B-style GEMM2, balanced routing.csv"))
olmoe_a100_skewed_routing = rename_columns(pandas.read_csv(olmoe_a100_dir / "OLMoE-1B-7B-style GEMM2, skewed routing.csv"))
olmoe_a100_perfect_routing.plot(x="num_tokens", ax=ax[0], legend=False)
olmoe_a100_balanced_routing.plot(x="num_tokens", ax=ax[1], legend=False)
olmoe_a100_skewed_routing.plot(x="num_tokens", ax=ax[2], legend=False)

ax[0].set_ylabel("TFLOP/s")
ax[0].set_xlabel("Number of Tokens")
ax[1].set_xlabel("Number of Tokens")
ax[2].set_xlabel("Number of Tokens")

plt.legend()
plt.savefig("m_grouped_time_plot_olmoe_a100.pdf")
plt.close()

# olmoe on H100 NVL
olmoe_h100_dir = pathlib.Path("moe-explore-results-h100/microbenchmarks/m_grouped_gemm/NVIDIA H100 NVL/1761700440.2513993/")
fix, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
olmoe_h100_perfect_routing = rename_columns(pandas.read_csv(olmoe_h100_dir / "OLMoE-1B-7B-style GEMM2, perfect routing.csv"))
olmoe_h100_balanced_routing = rename_columns(pandas.read_csv(olmoe_h100_dir / "OLMoE-1B-7B-style GEMM2, balanced routing.csv"))
olmoe_h100_skewed_routing = rename_columns(pandas.read_csv(olmoe_h100_dir / "OLMoE-1B-7B-style GEMM2, skewed routing.csv"))    
olmoe_h100_perfect_routing.plot(x="num_tokens", ax=ax[0], legend=False)
olmoe_h100_balanced_routing.plot(x="num_tokens", ax=ax[1], legend=False)
olmoe_h100_skewed_routing.plot(x="num_tokens", ax=ax[2])

ax[0].set_ylabel("TFLOP/s")
ax[0].set_xlabel("Number of Tokens")
ax[1].set_xlabel("Number of Tokens")
ax[2].set_xlabel("Number of Tokens")

plt.legend()
plt.savefig("m_grouped_time_plot_olmoe_h100.pdf")
plt.close()