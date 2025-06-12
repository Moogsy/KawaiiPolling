import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def summary_ratings_per_category(df: pd.DataFrame, ncols: int = 2):
    grouped = df.groupby(['Category', 'Rating'])
    metrics = grouped['Score'].agg(
        Count = 'count',
        Average = 'mean',
        Median = 'median',
        StdDev = lambda x: x.std(ddof=0),
        Min = 'min',
        Max = 'max'
    )

    means = metrics['Average'].unstack('Rating') # type: ignore
    stds = metrics['StdDev'].unstack('Rating') # type: ignore
    medians = metrics['Median'].unstack('Rating') # type: ignore
    mins = metrics['Min'].unstack('Rating') # type: ignore
    maxs = metrics['Max'].unstack('Rating') # type: ignore

    categories = means.index.tolist()
    ratings = means.columns.tolist()
    n_ratings = len(ratings)
    nrows = int(np.ceil(n_ratings / ncols))
    x = np.arange(len(categories))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), squeeze=False)

    cmap = plt.get_cmap('Set3', n_ratings)

    median_color = "Orange"
    max_color = "Red"
    min_color = "Blue"

    for idx, rating in enumerate(ratings):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        color = cmap(idx)
        ax.bar(x, means[rating], width=0.6, color=color)
        ax.errorbar(x, means[rating], yerr=stds[rating], fmt='none', ecolor='gray', capsize=3)
        ax.scatter(x, medians[rating], marker='D', color=median_color)
        ax.scatter(x, maxs[rating], marker='^', color=max_color)
        ax.scatter(x, mins[rating], marker='v', color=min_color)

        ax.set_title(rating)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylim(0.8, 5.2)
        ax.set_ylabel('Score')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Hide empty subplots
    for empty_idx in range(n_ratings, nrows*ncols):
        row, col = divmod(empty_idx, ncols)
        axes[row][col].axis('off')

    # Common labels
    fig.text(0.5, 0.04, 'Category', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical', fontsize=14)

    # Global legend
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='Median', markerfacecolor=median_color, markersize=8),
        Line2D([0], [0], marker='^', color='w', label='Max', markerfacecolor=max_color, markersize=8),
        Line2D([0], [0], marker='v', color='w', label='Min', markerfacecolor=min_color, markersize=8)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # type: ignore
    plt.show()

    return metrics


