r"""
Plots output data from `bench/bench_glu.py`
"""

import argparse
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()

files = glob.glob(f"{args.dir}/*.csv")
print(files)
for file in files:
    print(file)
    df = pd.read_csv(file)
    print(df)

    df["ScatterMoE"] = df["Fused MoE"] / df["ScatterMoE"]
    df["Torch Grouped MM"] = df["Fused MoE"] / df["Torch Grouped MM"]
    if "Huggingface" in df.keys():
        df["Huggingface"] = df["Fused MoE"] / df["Huggingface"]

    # This needs to be done last!
    df["Fused MoE"] = df["Fused MoE"] / df["Fused MoE"]
    df["seq_len"] = df["seq_len"].astype(int)

    melted_df = pd.melt(df, id_vars="seq_len", var_name="moe", value_name="time")
    seaborn.catplot(data=melted_df, x="seq_len", y="time", hue="moe", kind="bar", errorbar=None, legend_out=False)
    
    plt.ylabel("Percent of Fastest Runtime")
    plt.xlabel("Number of Tokens")    
    
    out_file = f"{args.dir}/{file.split('/')[-1].split('.')[0]}.png"
    print(f"saving fig to {out_file}")
    plt.savefig(out_file)
    plt.close()