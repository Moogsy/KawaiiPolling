import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

__all__ = ("plot_ceiling_and_floor_heatmaps",)

def get_ceiling_floor_proportions_all(df):
    """
    Calculate floor and ceiling proportions for all scales and all categories in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Category', 'Rating', 'Score']
    
    Returns:
        pd.DataFrame with columns ['Scale', 'Category', 'Floor Proportion', 'Ceiling Proportion']
    """
    results = []
    scales = df['Rating'].unique()
    categories = df['Category'].unique()
    
    for scale in scales:
        for category in categories:
            subset = df[(df['Rating'] == scale) & (df['Category'] == category)]
            total = len(subset)
            if total == 0:
                continue
            
            floor_prop = (subset['Score'] == 1).sum() / total
            ceil_prop  = (subset['Score'] == 5).sum() / total
            
            results.append({
                'Scale': scale,
                'Category': category,
                'Floor Proportion': floor_prop,
                'Ceiling Proportion': ceil_prop
            })
    
    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_ceiling_floor_effects(df: pd.DataFrame):
    """
    Plot ceiling and floor proportions in subplots arranged in a grid, one subplot per category.
    Each subplot shows all scales with floor and ceiling bars.
    
    Parameters:
        df_proportions (pd.DataFrame): Must have columns ['Scale', 'Category', 'Floor Proportion', 'Ceiling Proportion']
    """
    df_proportions = get_ceiling_floor_proportions_all(df)
    categories = df_proportions['Category'].unique()
    n_cats = len(categories)

    # Determine grid size (try to get as square as possible)
    n_cols = math.ceil(math.sqrt(n_cats))
    n_rows = math.ceil(n_cats / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes = axes.flatten()  # Flatten to easily iterate even if grid is 1D

    for ax_idx, ax in enumerate(axes):
        if ax_idx < n_cats:
            category = categories[ax_idx]
            data = df_proportions[df_proportions['Category'] == category]
            data_melted = data.melt(id_vars=['Scale'], value_vars=['Floor Proportion', 'Ceiling Proportion'],
                                    var_name='Effect', value_name='Proportion')

            sns.barplot(data=data_melted, x='Scale', y='Proportion', hue='Effect', ax=ax)
            ax.set_title(f"{category}")
            ax.set_ylim(0, 0.5)
            ax.set_xlabel('Scale')
            ax.set_ylabel('Proportion of Ratings')
            ax.tick_params(axis='x', rotation=45)
            if ax_idx == 0:
                ax.legend(title='Effect')
            else:
                ax.get_legend().remove()
        else:
            # Hide any unused subplots
            ax.axis('off')

    plt.suptitle('Ceiling and Floor Effects by Category and Scale', fontsize=16)
    plt.show()


def plot_ceiling_and_floor_heatmaps(df):
    """
    Plot ceiling and floor proportions side by side as heatmaps with different color scales.
    
    Parameters:
        df_proportions (pd.DataFrame): Must have columns ['Scale', 'Category', 'Floor Proportion', 'Ceiling Proportion']
    """
    df_proportions = get_ceiling_floor_proportions_all(df)
    # Pivot data for heatmaps
    ceiling_data = df_proportions.pivot(index='Category', columns='Scale', values='Ceiling Proportion')
    floor_data = df_proportions.pivot(index='Category', columns='Scale', values='Floor Proportion')

    _, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Ceiling heatmap (red)
    sns.heatmap(ceiling_data, annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': 'Ceiling Proportion'}, ax=axes[0])
    axes[0].set_title('Ceiling Proportion by Category and Scale')
    axes[0].set_ylabel('Category')
    axes[0].set_xlabel('Scale')

    # Floor heatmap (blue)
    sns.heatmap(floor_data, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Floor Proportion'}, ax=axes[1])
    axes[1].set_title('Floor Proportion by Category and Scale')
    axes[1].set_xlabel('Scale')

    plt.suptitle('Ceiling and Floor Effects by Category and Scale', fontsize=16)
    plt.show()

