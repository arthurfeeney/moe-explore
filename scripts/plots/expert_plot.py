import torch
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.constrained_layout.use'] = True

seaborn.set_theme(font_scale=1.1)

df = pd.read_csv("Qwen3-30B-A3B_experts_384.csv")
seaborn.lineplot(x="num_experts", y="Fused MoE", data=df, label="Fused MoE")
seaborn.lineplot(x="num_experts", y="ScatterMoE", data=df, label="ScatterMoE")

df = pd.read_csv("Qwen3-30B-A3B_experts_768.csv")
seaborn.lineplot(x="num_experts", y="Fused MoE", data=df, label="Fused MoE")
seaborn.lineplot(x="num_experts", y="ScatterMoE", data=df, label="ScatterMoE")

df = pd.read_csv("Qwen3-30B-A3B_experts_1536.csv")
seaborn.lineplot(x="num_experts", y="Fused MoE", data=df, label="Fused MoE")
seaborn.lineplot(x="num_experts", y="ScatterMoE", data=df, label="ScatterMoE")

plt.ylabel("Time (ms)")
plt.xlabel("Number of Experts")

plt.legend()
plt.savefig("expert_plot.png", transparent=True)