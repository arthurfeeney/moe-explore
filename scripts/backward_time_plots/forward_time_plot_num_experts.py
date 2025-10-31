import torch
import pandas
import seaborn
import matplotlib.pyplot as plt
import pathlib

seaborn.set_theme(
    font_scale=1.5, 
    style="white",
    palette=None,
    rc={"figure.constrained_layout.use": True}
)

def update(df):
    r"""
    This normalizes the columns to be speedups/slowdowns relative to Fused MoE.
    """
    df["Torch Grouped MM"] = df["Fused MoE"] / df["Torch Grouped MM"]
    if "Huggingface" in df.keys():
        del df["Huggingface"]# = df["Fused MoE"] / df["Huggingface"]
    df["ScatterMoE"] = df["Fused MoE"] / df["ScatterMoE"]
    df["Fused MoE"] = df["Fused MoE"] / df["Fused MoE"]
    df = df.rename(columns={
        "Torch Grouped MM": "Torch MoE",
        "Fused MoE": "Alloy MoE (Ours)"
    })
    
    df["num_experts"] = df["num_experts"].astype(int)
    
    return df

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_a100_num_experts_dir = pathlib.Path("results-backward/expert_counts/NVIDIA A100 80GB PCIe/1761638287.8716147/")
qwen3_a100_experts_384_df = update(pandas.read_csv(qwen3_a100_num_experts_dir / "Qwen3-30B-A3B_experts_hidden=384.csv"))
qwen3_a100_experts_768_df = update(pandas.read_csv(qwen3_a100_num_experts_dir / "Qwen3-30B-A3B_experts_hidden=768.csv"))
qwen3_a100_experts_384_df.plot.bar(x="num_experts", ax=ax[0], legend=False)
qwen3_a100_experts_768_df.plot.bar(x="num_experts", ax=ax[1], legend=False)
ax[0].set_ylabel("Relative Performance")
ax[0].set_xlabel("# of Experts, Intermediate-dim=384")
ax[1].set_xlabel("# of Experts, Intermediate-dim=768")
plt.savefig("backward_time_plot_qwen3_a100_num_experts.pdf")
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_h100_num_experts_dir = pathlib.Path("moe-explore-results-h100/results-backward/expert_counts/NVIDIA H100 NVL/1761698802.5129714/")
qwen3_h100_experts_384_df = update(pandas.read_csv(qwen3_h100_num_experts_dir / "Qwen3-30B-A3B_experts_hidden=384.csv"))
qwen3_h100_experts_768_df = update(pandas.read_csv(qwen3_h100_num_experts_dir / "Qwen3-30B-A3B_experts_hidden=768.csv"))
qwen3_h100_experts_384_df.plot.bar(x="num_experts", ax=ax[0], legend=False)
qwen3_h100_experts_768_df.plot.bar(x="num_experts", ax=ax[1])
ax[0].set_ylabel("Relative Performance")
ax[0].set_xlabel("# of Experts, Intermediate-dim=384")
ax[1].set_xlabel("# of Experts, Intermediate-dim=768")
plt.savefig("backward_time_plot_qwen3_h100_num_experts.pdf")
plt.close()