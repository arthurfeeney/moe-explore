import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from typing import Optional

plt.rcParams['figure.constrained_layout.use'] = True
seaborn.set_theme(font_scale=1.1)

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load-pt", type=str, required=True)
args = parser.parse_args()

data = torch.load(args.load_pt)

print("Visualizing:")
print(data)

expert_counts: torch.Tensor = data["expert_counts"]
num_tokens: int = data["num_tokens"]
model_name: Optional[str] = data.get("model_name")
dataset_name: Optional[str] = data.get("dataset_name")

expert_counts = expert_counts.numpy()
expert_percentages = 100 * (expert_counts / num_tokens)

for i in range(0, expert_percentages.shape[0], 10):
    print(f"Percentaages for layer {i}: {expert_percentages[i].tolist()}")

ax = seaborn.heatmap(expert_percentages, cbar_kws={"label": "Routing Percentages"})
ax.set(xlabel="Expert", ylabel="Layer")
colorbar = ax.collections[0].colorbar
assert colorbar is not None
colorbar_ticks = np.linspace(0, expert_percentages.max(), 4).astype(int).tolist()
colorbar_ticks = colorbar_ticks[:-1]
colorbar_ticks.append(expert_percentages.max())
print(colorbar_ticks)
colorbar.set_ticks(colorbar_ticks)
colorbar.set_ticklabels([f"{c:.0f}%" for c in colorbar_ticks])

fig_name_parts = []
if dataset_name is not None:
    fig_name_parts.append(dataset_name)
if model_name is not None:
    fig_name_parts.append(model_name)
fig_name_parts.append("percentages")
fig_name = f"{'_'.join(fig_name_parts)}.png"
print(f"Saving to {fig_name}")

plt.savefig(fig_name, transparent=True)

for idx in range(expert_percentages.shape[0]):
    min_perc = expert_percentages[idx].min()
    max_perc = expert_percentages[idx].max()
    print(f"layer {idx}: min={min_perc}, max={max_perc}")

