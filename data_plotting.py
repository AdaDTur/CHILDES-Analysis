import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

group_colors = {
    "zeros": "skyblue",
    "ones": "orange",
    "twos": "green",
    "threes": "red",
    "fours": "purple"
}

files_by_group = {
    "zeros": [
        "soderstrom.parsed_trees.txt"
    ],
    "ones": [
        "brown-eve+animacy+theta.parsed_trees.txt",
        "bernstein-wh.parsed_trees.txt"
    ],
    "twos": [
        "vanhouten-twos-wh.parsed_trees.txt",
        "valian+animacy+theta.parsed_trees.txt"
    ],
    "threes": [
        "vanhouten-threes-wh.parsed_trees.txt",
        "vankleeck-wh.parsed_trees.txt",
        "brown-adam3to4+animacy+theta.parsed_trees.txt",
    ],
    "fours": [
        "brown-adam4up+animacy+theta.parsed_trees.txt"
    ]
}

files_data = pd.read_json('childes_decomp.json').sort_values("group")

labels = ["Soderstrom", "Brown-Eve", "Bernstein-WH", "VanHouten-Twos-WH", "Valian", "VanHouten-Threes-WH", "VanKleeck", "Brown-Adam-1", "Brown-Adam-2"]

mlu_values = files_data['mlu']
avg_depth_values = files_data['avg_depth']
prop_sbar_values = files_data['sbar_stats']
ttr_values = files_data['ttr']
wh_values = files_data['wh_phrases']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axs = axs.flatten()

x_positions = np.arange(len(files_data))

for i, r in files_data.iterrows():
    color = group_colors[r["group"]]
    axs[0].bar(x_positions[i], r["mlu"], color=color)
axs[0].set_title("MLU (words)")
axs[0].set_xticks(x_positions)
axs[0].set_xticklabels(labels, rotation=90)

for i, r in files_data.iterrows():
    color = group_colors[r["group"]]
    axs[1].bar(x_positions[i], r["avg_depth"], color=color)
axs[1].set_title("Average Tree Depth")
axs[1].set_xticks(x_positions)
axs[1].set_xticklabels(labels, rotation=90)

for i, r in files_data.iterrows():
    color = group_colors[r["group"]]
    prop_val = r["sbar_stats"]["proportion_of_utterances_with_sbar"]
    axs[2].bar(x_positions[i], prop_val, color=color)
axs[2].set_title("Proportion of Utterances w/ SBAR")
axs[2].set_xticks(x_positions)
axs[2].set_xticklabels(labels, rotation=90)

for i, r in files_data.iterrows():
    color = group_colors[r["group"]]
    axs[3].bar(x_positions[i], r["ttr"], color=color)
axs[3].set_title("Type-Token Ratio")
axs[3].set_xticks(x_positions)
axs[3].set_xticklabels(labels, rotation=90)

for i, r in files_data.iterrows():
    color = group_colors[r["group"]]
    axs[4].bar(x_positions[i], r["wh_phrases"], color=color)
axs[4].set_title("Total WH-Phrases")
axs[4].set_xticks(x_positions)
axs[4].set_xticklabels(labels, rotation=90)

axs[5].axis('off')

plt.tight_layout()
plt.show()
