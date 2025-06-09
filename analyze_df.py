from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

def load_df_from_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=[0, 1, 2, 3], low_memory=False)

def gaussian_scores(df: pd.DataFrame):
    """
    Generates a histogram of all scores (centered 1–5 bins),
    fits a Gaussian curve to it, and displays the figure
    with x-axis spanning 0 to 6.
    """
    # 1. Extract data and compute fit parameters
    data = df["Score"].dropna()
    mu, sigma = np.mean(data), np.std(data, ddof=0)

    # 2. Create the plot
    _, ax = plt.subplots()

    # centered bins for 1,2,...,5
    bins = [i - 0.5 for i in range(1, 7)]
    _, edges, _ = ax.hist(
        data,
        bins=bins,
        edgecolor="black",
        alpha=0.6
    )

    # 3. Build and overlay the Gaussian PDF (scaled to histogram)
    x = np.linspace(edges[0], edges[-1], 200)
    bin_width = edges[1] - edges[0]
    y = norm.pdf(x, mu, sigma) * len(data) * bin_width

    ax.plot(
        x, y,
        lw=2,
        label=f"Gaussian fit\nμ={mu:.2f}, σ={sigma:.2f}"
    )

    # 4. Extend x-axis out to 0 and 6, but keep ticks at 1–5
    ax.set_xlim(0, 6)
    ax.set_xticks([1, 2, 3, 4, 5])

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Overall Score Distribution with Gaussian Fit")
    ax.legend()

    # 5. Display interactively
    plt.show()

def metrics_per_category(df: pd.DataFrame):
    """
    Generates a table plot of metrics for each Category and Rating kind:
    count, average, median, standard deviation, min, max.
    Also returns the metrics DataFrame.
    """
    grouped = df.groupby(['Category', 'Rating'])
    metrics = grouped['Score'].agg([
        ('Count', 'count'),
        ('Average', 'mean'),
        ('Median', 'median'),
        ('StdDev', lambda x: x.std(ddof=0)),
        ('Min', 'min'),
        ('Max', 'max')
    ])

    # Prepare table display
    # Combine MultiIndex into row labels
    row_labels = [f"{cat} - {rating}" for cat, rating in metrics.index]
    cell_text = np.round(metrics.values, 2).tolist() # type: ignore

    # Calculate figure size: width fixed, height based on number of rows
    n_rows = len(row_labels)
    fig_height = max(2, n_rows * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=metrics.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    plt.title("Metrics per Category and Rating")
    plt.show()

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
    corr = pivot.corr()
    return corr

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

    ax.set_title('Correlation Matrix of Rating Types', pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_df_from_csv("all_ratings.csv")
    # gaussian_scores(df)
    # metrics_per_category(df)
    plot_correlation_matrix(df)

