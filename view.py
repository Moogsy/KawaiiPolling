import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataframe(path: str = 'all_ratings.csv') -> pd.DataFrame:
    """
    Load the CSV back into the same MultiIndex DataFrame,
    and rename its lone data column to 'Score'.
    """
    df = pd.read_csv(path, index_col=[0,1,2,3], low_memory=False)
    df.columns = ['Score']
    df.index.names = ['Category', 'Pose', 'Rating', 'RaterID']
    return df

def plot_overall_distribution(df: pd.DataFrame):
    """Histogram of all scores."""
    data = df['Score'].dropna()
    fig, ax = plt.subplots()
    ax.hist(data, bins=10)
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title('Overall Score Distribution')
    plt.show()

def plot_summary_stats(df: pd.DataFrame):
    """
    Bar chart of mean ± std for each (Category, Rating) pair.
    """
    flat = df.reset_index()
    summary = flat.groupby(['Category','Rating'])['Score'] \
                  .agg(['mean','std']) \
                  .unstack('Rating')
    means = summary['mean']
    stds = summary['std']

    categories = means.index
    ratings = means.columns
    x = np.arange(len(categories))
    width = 0.8 / len(ratings)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, rating in enumerate(ratings):
        ax.bar(
            x + i * width,
            means[rating],
            width,
            yerr=stds[rating],
            capsize=5,
            label=rating
        )
    ax.set_xticks(x + width * (len(ratings) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Score by Category and Rating (±1 SD)')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_coefficient_of_variation(df: pd.DataFrame):
    """
    Grouped bar chart of CoV (std/mean) for each (Category, Rating).
    """
    flat = df.reset_index()
    summary = flat.groupby(['Category','Rating'])['Score'] \
                  .agg(['mean','std'])
    cov = (summary['std'] / summary['mean']).unstack('Rating')

    categories = cov.index
    ratings = cov.columns
    x = np.arange(len(categories))
    width = 0.8 / len(ratings)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, rating in enumerate(ratings):
        ax.bar(
            x + i * width,
            cov[rating],
            width,
            label=rating
        )
    ax.set_xticks(x + width * (len(ratings) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Coefficient of Variation by Category and Rating')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pose_variance_by_rater(df: pd.DataFrame):
    """
    Bar chart of pose-to-pose score variance for each rater.
    """
    mean_by_pose = df['Score'].groupby(['RaterID','Pose']).mean()
    pivot = mean_by_pose.unstack('Pose')
    var_series = pivot.var(axis=1).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    var_series.plot.bar(ax=ax)
    ax.set_ylabel('Variance of Mean Scores Across Poses')
    ax.set_title('Pose-to-Pose Score Variance by Rater')
    plt.tight_layout()
    plt.show()

def plot_inter_rating_correlation(df: pd.DataFrame):
    """
    Heatmap of inter-rating correlation matrix.
    """
    wide = df['Score'].unstack('Rating').dropna()
    corr = wide.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title('Inter-Rating Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_cronbach_alpha(df: pd.DataFrame):
    """
    Display Cronbach's alpha in a standalone figure.
    """
    items = df['Score'].unstack('RaterID')
    k = items.shape[1]
    item_vars = items.var(axis=1, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    alpha = (k/(k-1)) * (1 - item_vars.sum() / total_var) # type: ignore

    fig, ax = plt.subplots()
    ax.text(
        0.5, 0.5,
        f"Cronbach's α = {alpha:.3f}",
        ha='center', va='center', fontsize=16
    )
    ax.axis('off')
    plt.title("Inter-Rater Reliability")
    plt.show()

if __name__ == '__main__':
    # Load
    df = load_dataframe('all_ratings.csv')

    # Visualizations
    plot_overall_distribution(df)
    plot_summary_stats(df)
    plot_coefficient_of_variation(df)
    plot_pose_variance_by_rater(df)
    plot_inter_rating_correlation(df)
    plot_cronbach_alpha(df)


