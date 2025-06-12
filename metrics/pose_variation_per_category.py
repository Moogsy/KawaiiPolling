import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = ["plot_kruskal_significance_logscale"]

def kruskal_test(df, alpha=0.05):
    categories = sorted(df['Category'].unique())
    ratings = sorted(df['Rating'].unique())

    results = []
    for category in categories:
        for rating in ratings:
            subset = df[(df['Category'] == category) & (df['Rating'] == rating)]
            groups = [group['Score'].values for _, group in subset.groupby('Pose')]
            if len(groups) > 1:
                stat, p = stats.kruskal(*groups)
            else:
                stat, p = np.nan, np.nan
            results.append({
                'Category': category,
                'Rating': rating,
                'Kruskal_stat': stat,
                'p_value': p,
                'Significant': (p < alpha) if not np.isnan(p) else False
            })
    return pd.DataFrame(results)

def format_pvalue_with_stars(p):
    if np.isnan(p):
        return "?"
    elif p < 0.001:
        return f"{p:.3f}"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"

def plot_kruskal_significance_logscale(df: pd.DataFrame, alpha=0.05):
    results_df = kruskal_test(df, alpha)
    heatmap_data = results_df.pivot(index='Category', columns='Rating', values='p_value')

    # Replace zeros or NaNs to avoid -inf in log scale
    heatmap_data = heatmap_data.replace(0, 1e-10)
    heatmap_data = heatmap_data.fillna(1)  # treat NaN as max p-value (non-significant)

    # Transform p-values to -log10(p) for color scale (higher means more significant)
    log_heatmap_data = -np.log10(heatmap_data)

    # Format annotations with original p-values and stars
    annotations = heatmap_data.applymap(format_pvalue_with_stars)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    cmap = sns.light_palette("red", as_cmap=True)

    sns.heatmap(
        log_heatmap_data,
        cmap=cmap,
        annot=annotations,
        fmt='',
        linewidths=0.5,
        linecolor='gray',
        square=True,
        cbar_kws={'label': '-log10(p-value)'},
        annot_kws={"fontsize": 9}
    )

    plt.title("Significance of Pose Differences within Categories (Kruskal-Wallis)\n(log scale coloring)", fontsize=14)
    plt.suptitle(
        "H0: All poses within a category have the same mean rating on the scale\n"
        "H1: At least one pose differs significantly in mean rating within the category",
        fontsize=10, fontstyle='italic', ha='center', x=0.65
    )
    plt.xlabel("Rating Scale")
    plt.ylabel("Category")

    plt.show()

