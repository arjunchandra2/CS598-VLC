import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    data = pd.read_csv("./results/benchmark.csv")
    data = data.fillna("")
    data["Dataset"] = (data["data"]+"_"+data["category"]).apply(lambda x: x.rstrip("_"))
    data = data[data["data"] != "sugarcrepe"]

    # get excess loss
    data["acc_baseline"] = .5
    data.loc[(data["data"] == "aro_coco"), 'acc_baseline'] = .2
    data.loc[(data["data"] == "aro_flickr"), 'acc_baseline'] = .2
    data["Accuracy Lift"] = data["acc_uni"] - data["acc_baseline"]



    ax = sns.barplot(data=data, x="Dataset", y="Accuracy Lift", hue="model", edgecolor="black")

    # Define patterns for different hue categories
    hatch_patterns = {
        "DAC-SAM": "",   # Diagonal pattern
        "ViT": "//"    # Cross pattern
    }

    # Define color map
    palette = sns.color_palette()
    color_map = {x: palette[i] for i, x in enumerate(data["data"].unique())}

    
    # Apply colors and patterns manually
    for bars, hue_category in zip(ax.containers, data["model"].unique()):
        for bar, (_, row) in zip(bars, data.iterrows()):
            bar.set_facecolor(color_map[row["data"]])   # Apply color based on 'color_by'
            bar.set_hatch(hatch_patterns[hue_category])  # Apply pattern based on hue


    # Adjust legend to include hatching
    legend = ax.legend(title="Model")
    for legend_patch, hue_category in zip(legend.get_patches(), data["model"].unique()):
        legend_patch.set_facecolor("white")
        legend_patch.set_hatch(hatch_patterns[hue_category])

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.savefig("./results/benchmark.pdf", bbox_inches='tight')