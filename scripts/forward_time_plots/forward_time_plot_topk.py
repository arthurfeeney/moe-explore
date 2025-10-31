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
    del df["Huggingface"]
    #df["Huggingface"] = df["Fused MoE"] / df["Huggingface"]
    df["ScatterMoE"] = df["Fused MoE"] / df["ScatterMoE"]
    df["Fused MoE"] = df["Fused MoE"] / df["Fused MoE"]
    df = df.rename(columns={
        "Torch Grouped MM": "Torch MoE",
        "Fused MoE": "Alloy MoE (Ours)"
    })
    
    df["act_experts"] = df["act_experts"].astype(int)
    
    return df

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_a100_topk_dir = pathlib.Path("results-forward/topk/NVIDIA A100 80GB PCIe/1761637347.120457/")
qwen3_a100_topk_384_df = update(pandas.read_csv(qwen3_a100_topk_dir / "Qwen3-30B-A3B_topk_hidden=384.csv"))
qwen3_a100_topk_512_df = update(pandas.read_csv(qwen3_a100_topk_dir / "Qwen3-30B-A3B_topk_hidden=512.csv"))
qwen3_a100_topk_768_df = update(pandas.read_csv(qwen3_a100_topk_dir / "Qwen3-30B-A3B_topk_hidden=768.csv"))

qwen3_a100_topk_384_df.plot.bar(x="act_experts", ax=ax[0], legend=False, width=0.9)
qwen3_a100_topk_512_df.plot.bar(x="act_experts", ax=ax[1], legend=False, width=0.9)
qwen3_a100_topk_768_df.plot.bar(x="act_experts", ax=ax[2], legend=False, width=0.9)
ax[0].set_ylabel("Relative Performance")
ax[0].set_xlabel("Top-k, Int. dim=384")
ax[1].set_xlabel("Top-k, Int. dim=512")
ax[2].set_xlabel("Top-k, Int. dim=768")
plt.savefig("forward_time_plot_qwen3_a100_topk.pdf")
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4), constrained_layout=True)
qwen3_h100_topk_dir = pathlib.Path("moe-explore-results-h100/results-forward/topk/NVIDIA H100 NVL/1761700110.0535007/")
qwen3_h100_topk_384_df = update(pandas.read_csv(qwen3_h100_topk_dir / "Qwen3-30B-A3B_topk_hidden=384.csv"))
qwen3_h100_topk_512_df = update(pandas.read_csv(qwen3_h100_topk_dir / "Qwen3-30B-A3B_topk_hidden=512.csv"))
qwen3_h100_topk_768_df = update(pandas.read_csv(qwen3_h100_topk_dir / "Qwen3-30B-A3B_topk_hidden=768.csv"))
qwen3_h100_topk_384_df.plot.bar(x="act_experts", ax=ax[0], legend=False, width=0.9)
qwen3_h100_topk_512_df.plot.bar(x="act_experts", ax=ax[1], legend=False, width=0.9)
qwen3_h100_topk_768_df.plot.bar(x="act_experts", ax=ax[2], width=0.9)
ax[0].set_ylabel("Relative Performance")
ax[0].set_xlabel("Top-k, Int. dim=384")
ax[1].set_xlabel("Top-k, Int. dim=512")
ax[2].set_xlabel("Top-k, Int. dim=768")
plt.savefig("forward_time_plot_qwen3_h100_topk.pdf")
plt.close()