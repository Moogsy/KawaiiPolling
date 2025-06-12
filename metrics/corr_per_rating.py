import pandas as pd
import numpy as np
import pingouin as pg
from matplotlib import pyplot as plt

__all__ = ("plot_correlation_matrix",)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the correlation matrix of scores across all Rating types.
    Pivoting on ['Category', 'Pose', 'RaterID'] as index and 'Rating' as columns.
    Returns a DataFrame where both rows and columns are rating kinds.
    """
    pivot = df.pivot_table(
        index=['Category', 'Pose', 'RaterID'],
        columns='Rating',
        values='Score'
    )

    return pivot.corr("spearman")


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Computes and displays the correlation matrix of rating types in a heatmap,
    annotated with the correlation values.
    """
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)

    # Annotate each cell with the numeric value
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}",
                    ha='center', va='center', color='black')

    ax.set_title('Correlation Matrix of Rating Types (spearman correlation)', pad=20)
    plt.tight_layout()
    plt.show()

