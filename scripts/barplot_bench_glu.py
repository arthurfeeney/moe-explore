import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.constrained_layout.use'] = True

seaborn.set_theme(font_scale=1.1)

def barplot(csv_path, plot_name):
    df = pd.read_csv(csv_path)
    df = df.melt(id_vars=["seq_len"], value_vars=["Grouped GLU", "Fused GLU"])#, "ScatterMoE"])
    df["seq_len"] = df["seq_len"].astype(int)
    df = df.rename(columns={"seq_len": "Sequence Length", "value": "Time (ms)"})
    ax = seaborn.barplot(x="Sequence Length", y="Time (ms)", hue="variable", data=df)
    plt.savefig(plot_name)
    plt.close()

barplot("Qwen3-30B-A3B_experts=8_128.csv", "qwen3-perf-barplot.png")
barplot("OLMoE-1B-7B_experts=8_64.csv", "olmoe-perf-barplot.png")

def lineplot(csv_path, plot_name):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Grouped GLU": "Unfused MoE (Grouped GEMM)",
        "Fused GLU": "Fused MoE (Grouped GLU)"
    })
    df = df.melt(id_vars=["seq_len"], value_vars=["Unfused MoE (Grouped GEMM)", "Fused MoE (Grouped GLU)", "ScatterMoE"])
    df["seq_len"] = df["seq_len"].astype(int)
    df = df.rename(columns={"seq_len": "Sequence Length", "value": "Time (ms)"})
    ax = seaborn.lineplot(x="Sequence Length", y="Time (ms)", hue="variable", data=df)
    plt.savefig(plot_name)
    plt.close()

lineplot("Qwen3-30B-A3B_experts=8_128.csv", "qwen3-perf-lineplot.pdf")
lineplot("OLMoE-1B-7B_experts=8_64.csv", "olmoe-perf-lineplot.pdf")
lineplot("Ernie4.5_experts=6_64.csv", "ernie-perf-lineplot.pdf")

def plot_breakdown(plot_name):
    r"""
    This is disgusting, but temporary:
      these numbers are copied manually from a profile `bench/profile_functional.py` using
      the proton profiler.
    The proton profiler output a .hatchet file, and all values are from the same level.
    Should be easy to read from that...
    """
    qwen3_fused_breakdown = {
        "Router": 0.037,
        "Fused Grouped GLU": 1.032,
        "Get Token Indices": 0.052,
        "Grouped GEMM": 0.755,
    }
    qwen3_unfused_breakdown = {
        "Router": 0.037,
        "Input Permute": 0.059,
        "Gate Grouped Gemm": 0.742,
        "Up Grouped Gemm": 0.740,
        "Gating + Activation": 0.007 + 0.006,
        "Down Grouped GEMM": 0.748,
        "Output Permute": 0.133
    }
    
    #fused_times = np.array([0.037, 0.052, 1.032, 0.755])
    #unfused_times = np.array([0.037, 0.059, 0.742, 0.740, 0.007 + 0.006, 0.748, 0.133])
    
    #fused_times = np.array([0.237, 0.013, 17.307, 11.278, 3.296])
    #unfused_times = np.array([0.220, 0.174, 7.794, 0.946, 7.745, 1.431, 8.418, 7.993])
    
    fused_times = np.array([9.420, 1.882, 161.530, 110.926, 28.259])
    unfused_times = np.array([8.712, 27.822, 76.834, 9.446, 76.414, 14.345, 86.612, 318.952])
    
    seaborn.barplot(
        x=["Unfused MoE (Grouped GEMM)"]*len(unfused_times), 
        y=np.cumsum(unfused_times)[::-1],
        hue=list(range(len(unfused_times))),
        dodge=False,
        legend=False)
    ax = seaborn.barplot(
        x=["Fused MoE (Grouped GLU)"]*len(fused_times), 
        y=np.cumsum(fused_times)[::-1],
        hue=list(range(len(fused_times))),
        dodge=False,
        legend=False)
    ax.set(ylabel="Time (ms)")
    plt.savefig(plot_name, transparent=True)
    plt.close()

plot_breakdown("qwen3-perf-breakdown.png")