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

files_data = pd.read_json('childes_data_summary.json').sort_values("group")

#labels_non_wh = ["Soderstrom", "Brown-Eve", "Valian", "Brown-Adam-1", "Brown-Adam-2"]

#labels_wh = ["Bernstein-WH", "VanHouten-Twos-WH", "VanHouten-Threes-WH", "VanKleeck-WH"]

labels_full = ["Soderstrom", "Brown-Eve", "Bernstein-WH", "VanHouten-Twos-WH", "Valian", "VanHouten-Threes-WH", "VanKleeck-WH", "Brown-Adam-1", "Brown-Adam-2"]


labels = labels_full
x_positions = np.arange(len(files_data))

def plot_metric(metric_values, title, ylabel, filename):
    plt.figure(figsize=(12, 7))

    for i, r in files_data.iterrows():
        color = group_colors[r["group"]]
        plt.bar(x_positions[i], metric_values[i], color=color)

    plt.xticks(x_positions, labels, rotation=45, fontsize=14)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12)
               for color in group_colors.values()]
    labels_legend = list(group_colors.keys())
    plt.legend(handles, labels_legend, title="Groups", loc='upper right', fontsize=14, title_fontsize=16)

    plt.tight_layout()
    plt.savefig(filename)

plot_metric(files_data['mlu'], "MLU (words)", "Mean Length of Utterance", 'mlu.jpg')
plot_metric(files_data['avg_depth'], "Average Tree Depth", "Avg Depth", 'depth.jpg')
plot_metric([r["sbar_stats"]["proportion_of_utterances_with_sbar"] for _, r in files_data.iterrows()],
            "Proportion of Utterances w/ SBAR", "Proportion", 'sbar.jpg')
plot_metric(files_data['ttr'], "Type-Token Ratio", "Type-Token Ratio", 'ttr.jpg')
