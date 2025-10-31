r"""
Plots output data from `bench/bench_m_grouped_gemm.py`
"""

import argparse
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()

#model = "OLMoE-1B-7B-style GEMM2"
#model = "Qwen3-30B-A3B-style GEMM2"

files = glob.glob(f"{args.dir}/*.csv")
print(files) 

for file in files:

    df = pd.read_csv(f"{file}")
    print(df.keys())
    print(df)
    print(df['torch.bmm'].max())

    seaborn.lineplot(x="num_tokens", y="torch.bmm", data=df, label="torch.bmm")
    seaborn.lineplot(x="num_tokens", y="Torch Grouped MM", data=df, label="torch._grouped_mm")
    seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Grouped-only")
    seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Grouped+Gather")
    seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Grouped+Scatter")
    plt.ylabel("TFLOP/s")
    plt.xlabel("Number of Tokens")

    plt.legend()

    save_path = f"{args.dir}/{file.split('/')[-1].split('.')[0]}.png"
    print(f"saving fig to {save_path}")
    plt.savefig(save_path)
    plt.close()

    """
    df = pd.read_csv(f"{model}, balanced routing.csv")
    seaborn.lineplot(x="num_tokens", y="torch.bmm", data=df, label="torch.bmm")
    seaborn.lineplot(x="num_tokens", y="Torch Grouped MM", data=df, label="torch._grouped_mm")
    seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Grouped-only")
    seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Grouped+Gather")
    seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Grouped+Scatter")

    plt.ylabel("TFLOP/s")
    plt.xlabel("Number of Tokens")

    plt.legend()
    plt.savefig("grouped_gemm_time_plot_balanced.pdf")
    plt.close()

    df = pd.read_csv(f"{model}, skewed routing.csv")
    seaborn.lineplot(x="num_tokens", y="torch.bmm", data=df, label="torch.bmm")
    seaborn.lineplot(x="num_tokens", y="Torch Grouped MM", data=df, label="torch._grouped_mm")
    seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Grouped-only")
    seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Grouped+Gather")
    seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Grouped+Scatter")


    #df = pd.read_csv("Qwen3-30B-A3B-style GEMM, skewed routing.csv")
    #seaborn.lineplot(x="num_tokens", y="Grouped-only", data=df, label="Skewed-Grouped-only", dashes=True)
    #seaborn.lineplot(x="num_tokens", y="Grouped+Gather", data=df, label="Skewed-Grouped+Gather", dashes=True)
    #seaborn.lineplot(x="num_tokens", y="Grouped+Scatter", data=df, label="Skewed-Grouped+Scatter", dashes=True)
    """
    
    #plt.legend()
    #plt.savefig(f"{args.dir}/grouped_gemm_time_plot_skewed.pdf")