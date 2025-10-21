import torch
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.constrained_layout.use'] = True

seaborn.set_theme(font_scale=1.1)

df = pd.read_csv("OLMoE-1B-7B_experts=8_64.csv")
seaborn.lineplot(x="seq_len", y="Fused MoE", data=df, label="Fused MoE")
#seaborn.lineplot(x="seq_len", y="ScatterMoE", data=df, label="ScatterMoE")
seaborn.lineplot(x="seq_len", y="Torch Grouped MM", data=df, label="Torch Grouped MM")

plt.ylabel("Time (ms)")
plt.xlabel("Number of Tokens")

plt.legend()
plt.savefig("num_tokens_plot_olmoe.pdf", transparent=True)
plt.close()

df = pd.read_csv("Qwen3-30B-A3B_experts=8_128.csv")
seaborn.lineplot(x="seq_len", y="Fused MoE", data=df, label="Fused MoE")
#seaborn.lineplot(x="seq_len", y="ScatterMoE", data=df, label="ScatterMoE")
seaborn.lineplot(x="seq_len", y="Torch Grouped MM", data=df, label="Torch Grouped MM")

plt.ylabel("Time (ms)")
plt.xlabel("Number of Tokens")

plt.legend()
plt.savefig("num_tokens_plot_qwen3.pdf", transparent=True)