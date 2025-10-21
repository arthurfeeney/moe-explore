import pandas as pd
import seaborn
import matplotlib.pyplot as plt

model = "OLMoE-1B-7B-style GEMM2"
#model = "Qwen3-30B-A3B-style GEMM2"

df = pd.read_csv(f"{model}, perfect routing.csv")
seaborn.lineplot(x="num_tokens", y="Torch Grouped MM", data=df, label="torch._grouped-mm")
seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Grouped-only")
seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Grouped+Gather")
seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Grouped+Scatter")
plt.ylabel("TFLOP/s")
plt.xlabel("Number of Tokens")

plt.legend()
plt.savefig("grouped_gemm_time_plot_perfect.pdf")
plt.close()

df = pd.read_csv(f"{model}, balanced routing.csv")
seaborn.lineplot(x="num_tokens", y="Torch Grouped MM", data=df, label="torch._grouped-mm")
seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Grouped-only")
seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Grouped+Gather")
seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Grouped+Scatter")

plt.ylabel("TFLOP/s")
plt.xlabel("Number of Tokens")

plt.legend()
plt.savefig("grouped_gemm_time_plot_balanced.pdf")
plt.close()

df = pd.read_csv(f"{model}, skewed routing.csv")
seaborn.lineplot(x="num_tokens", y="Torch Grouped MM", data=df, label="torch._grouped-mm")
seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Grouped-only")
seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Grouped+Gather")
seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Grouped+Scatter")


#df = pd.read_csv("Qwen3-30B-A3B-style GEMM, skewed routing.csv")
#seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Skewed-Grouped-only", dashes=True)
#seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Skewed-Grouped+Gather", dashes=True)
#seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Skewed-Grouped+Scatter", dashes=True)

plt.ylabel("TFLOP/s")
plt.xlabel("Number of Tokens")

plt.legend()
plt.savefig("grouped_gemm_time_plot_skewed.pdf")