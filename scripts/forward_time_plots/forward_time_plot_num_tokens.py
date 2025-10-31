import torch
import pandas
import seaborn
import matplotlib.pyplot as plt
import pathlib

seaborn.set_theme(
    font_scale=1.4,
    style="white",
    palette=None,
    rc={"figure.constrained_layout.use": True}
)

def update(df):
    r"""
    This normalizes the columns to be speedups/slowdowns relative to Fused MoE.
    """
    df["Torch Grouped MM"] = df["Fused MoE"] / df["Torch Grouped MM"]
    #if "Huggingface" in df.keys():
    #    del df["Huggingface"]
    df["Huggingface"] = df["Fused MoE"] / df["Huggingface"]
    df["ScatterMoE"] = df["Fused MoE"] / df["ScatterMoE"]
    df["Fused MoE"] = df["Fused MoE"] / df["Fused MoE"]
    df = df.rename(columns={
        "Torch Grouped MM": "Torch MoE",
        "Fused MoE": "Alloy MoE (Ours)"
    })
    
    df["seq_len"] = df["seq_len"].astype(int)
    
    return df

a100_num_tokens_dir = pathlib.Path("results-forward/num_tokens/NVIDIA A100 80GB PCIe/1761636427.0253744/")
a100_num_tokens_qwen3_df = update(pandas.read_csv(a100_num_tokens_dir / "Qwen3-30B-A3B_experts=8_128.csv"))
a100_num_tokens_olmoe_df = update(pandas.read_csv(a100_num_tokens_dir / "OLMoE-1B-7B_experts=8_64.csv"))
a100_num_tokens_ernie4_df = update(pandas.read_csv(a100_num_tokens_dir / "Ernie4.5_experts=6_64.csv"))

a100_num_tokens_qwen3_df = a100_num_tokens_qwen3_df.melt(id_vars=["seq_len"], var_name="impl", value_name="time")
seaborn.barplot(data=a100_num_tokens_qwen3_df, x="seq_len", y="time", hue="impl", legend=False)
plt.ylabel("Relative Runtime")
plt.xlabel("Number of Tokens")
plt.savefig("forward_time_plot_a100_num_tokens_qwen3.pdf")
plt.close()
a100_num_tokens_olmoe_df = a100_num_tokens_olmoe_df.melt(id_vars=["seq_len"], var_name="impl", value_name="time")
seaborn.barplot(data=a100_num_tokens_olmoe_df, x="seq_len", y="time", hue="impl", legend=False)
plt.ylabel("Relative Runtime")
plt.xlabel("Number of Tokens")
plt.savefig("forward_time_plot_a100_num_tokens_olmoe.pdf")
plt.close()
a100_num_tokens_ernie4_df = a100_num_tokens_ernie4_df.melt(id_vars=["seq_len"], var_name="impl", value_name="time")
ax = seaborn.barplot(data=a100_num_tokens_ernie4_df, x="seq_len", y="time", hue="impl")
ax.legend_.set_title(None)
plt.ylabel("Relative Runtime")
plt.xlabel("Number of Tokens")
plt.savefig("forward_time_plot_a100_num_tokens_ernie4.pdf")
plt.close()

h100_num_tokens_dir = pathlib.Path("moe-explore-results-h100/1761808515.627374/")
h100_num_tokens_qwen3_df = update(pandas.read_csv(h100_num_tokens_dir / "Qwen3-30B-A3B_experts=8_128.csv"))
h100_num_tokens_olmoe_df = update(pandas.read_csv(h100_num_tokens_dir / "OLMoE-1B-7B_experts=8_64.csv"))
h100_num_tokens_ernie4_df = update(pandas.read_csv(h100_num_tokens_dir / "Ernie4.5_experts=6_64.csv"))

h100_num_tokens_qwen3_df = h100_num_tokens_qwen3_df.melt(id_vars=["seq_len"], var_name="impl", value_name="time")
seaborn.barplot(data=h100_num_tokens_qwen3_df, x="seq_len", y="time", hue="impl", legend=False)
plt.ylabel("Relative Performance")
plt.xlabel("Number of Tokens")
plt.legend()
plt.savefig("forward_time_plot_h100_num_tokens_qwen3.pdf")
plt.close()
h100_num_tokens_olmoe_df = h100_num_tokens_olmoe_df.melt(id_vars=["seq_len"], var_name="impl", value_name="time")
seaborn.barplot(data=h100_num_tokens_olmoe_df, x="seq_len", y="time", hue="impl", legend=False)
plt.ylabel("Relative Performance")
plt.xlabel("Number of Tokens")
plt.savefig("forward_time_plot_h100_num_tokens_olmoe.pdf")
plt.close()
h100_num_tokens_ernie4_df = h100_num_tokens_ernie4_df.melt(id_vars=["seq_len"], var_name="impl", value_name="time")
ax = seaborn.barplot(data=h100_num_tokens_ernie4_df, x="seq_len", y="time", hue="impl")
ax.legend_.set_title(None)
plt.ylabel("Relative Performance")
plt.xlabel("Number of Tokens")
plt.savefig("forward_time_plot_h100_num_tokens_ernie4.pdf")
plt.close()