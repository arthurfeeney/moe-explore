import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

data = torch.load("wikitext_counts_qwen3.pt")
print(data)

olmoe_percentages = data["expert_counts"] / data["num_tokens"]
olmoe_percentages, _ = olmoe_percentages.sort(dim=1)
olmoe_percentages = olmoe_percentages.cpu().numpy()

for i in range(olmoe_percentages.shape[0]):
    ax = seaborn.lineplot(x=np.arange(len(olmoe_percentages[i])), y=olmoe_percentages[i])
    ax.set(xlabel="Expert, Sorted by Routing Percentage", ylabel="Routing Percentage")
plt.savefig(f"olmoe_layer_probabilities.png")