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

files_data = pd.read_json('childes_decomp.json').sort_values("group")

labels = ["Soderstrom", "Brown-Eve", "Bernstein-WH", "VanHouten-Twos-WH", "Valian", 
          "VanHouten-Threes-WH", "VanKleeck", "Brown-Adam-1", "Brown-Adam-2"]

x_positions = np.arange(len(files_data))

def plot_metric(metric_values, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    for i, r in files_data.iterrows():
        color = group_colors[r["group"]]
        plt.bar(x_positions[i], metric_values[i], color=color)
    
    plt.xticks(x_positions, labels, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
               for color in group_colors.values()]
    labels_legend = list(group_colors.keys())
    plt.legend(handles, labels_legend, title="Groups", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)

plot_metric(files_data['mlu'], "MLU (words)", "Mean Length of Utterance", "mlu_plot.jpg")
plot_metric(files_data['avg_depth'], "Average Tree Depth", "Avg Depth", "tree_depth_plot.jpg")
plot_metric([r["sbar_stats"]["proportion_of_utterances_with_sbar"] for _, r in files_data.iterrows()], 
            "Proportion of Utterances w/ SBAR", "Proportion", "sbar_plot.jpg")
plot_metric(files_data['ttr'], "Type-Token Ratio", "Type-Token Ratio", "ttr_plot.jpg")
plot_metric(files_data['wh_phrases'], "Total WH-Phrases", "Total WH-Phrases", "wh_phrases_plot.jpg")
