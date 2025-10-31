import seaborn
import matplotlib.pyplot as plt
import pandas as pd

#plt.rcParams['figure.constrained_layout.use'] = True
seaborn.set_theme(
    font_scale=1.3, 
    style="whitegrid", 
    palette=None,
    rc={"figure.constrained_layout.use": True}
)

def dict_to_df(d):
    k, v = zip(*d.items())
    return pd.DataFrame({"batch_size": k, "throughput": v})
    
topk_moe_olmoe_throughputs = {
    # batch_size : tokens / second
    2: 27050,
    4: 30882,
    6: 32995,
    8: 34155,
    10: 34657,
    12: 35127,
    14: 35201,
    16: 35604,
    18: 35604,
    20: 35764,
    22: 35818,
    24: 36052,
    26: 35967,
    28: 36078,
    30: 36130,
    32: 36253
}
topk_moe_df = dict_to_df(topk_moe_olmoe_throughputs)

torch_moe_olmoe_throughputs = {
    # batch_size : tokens / second
    2: 26065,
    4: 30318,
    6: 31927,
    8: 33022,
    10: 33440,
    12: 33872,
    14: 33918,
    16: 34059,
    18: 34107,
    20: 34309,
    22: 34280,
    24: 34425,
    26: 34355,
    28: 34325
}

torch_moe_df = dict_to_df(torch_moe_olmoe_throughputs)

seaborn.lineplot(x="batch_size", y="throughput", data=topk_moe_df, label="Alloy MoE (Ours)")
seaborn.lineplot(x="batch_size", y="throughput", data=torch_moe_df, label="Torch MoE")
plt.ylabel("Tokens / Second")
plt.xlabel("Batch Size")
plt.legend()
plt.savefig("throughput_plot.pdf")
plt.close()