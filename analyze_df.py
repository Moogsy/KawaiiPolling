from os import walk
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import pingouin as pg
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
    mu, sigma = np.mean(data), np.std(data, ddof=1) # ddof=1 is better for small samples

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

def metrics_per_category(df: pd.DataFrame, ncols: int = 2):
    """
    Plots one subplot per Rating scale, showing for each Category:
      - bar    = mean Score
      - error  = ±1 StdDev
      - ◊      = Median
      - ▲      = Max
      - ▼      = Min
    Returns the metrics DataFrame.
    """
    # 1) compute metrics
    grouped = df.groupby(['Category', 'Rating'])
    metrics = grouped['Score'].agg(
        Count = 'count',
        Average = 'mean',
        Median = 'median',
        StdDev = lambda x: x.std(ddof=0),
        Min = 'min',
        Max = 'max'
    )

    # 2) pivot into wide tables
    means   = metrics['Average'].unstack('Rating') # type: ignore
    stds    = metrics['StdDev'].unstack('Rating') # type: ignore
    medians = metrics['Median'].unstack('Rating') # type: ignore
    mins    = metrics['Min'].unstack('Rating') # type: ignore
    maxs    = metrics['Max'].unstack('Rating') # type: ignore

    categories = means.index.tolist()
    ratings    = means.columns.tolist()
    n_ratings  = len(ratings)
    nrows      = int(np.ceil(n_ratings / ncols))
    x          = np.arange(len(categories))

    # 3) make subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False
    )

    # 4) plot each rating in its own ax
    for idx, rating in enumerate(ratings):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # bars + errorbars
        ax.bar(x, means[rating], width=0.6)
        ax.errorbar(x, means[rating], yerr=stds[rating], fmt='none', ecolor='pink', capsize=4)

        # markers
        ax.scatter(x, medians[rating], marker='D', label='Median')
        ax.scatter(x, maxs[rating],    marker='^', label='Max')
        ax.scatter(x, mins[rating],    marker='v', label='Min')

        # decorations
        ax.set_title(rating)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylim(0.8, 5.2)
        ax.set_ylabel('Score')

        # legend (only markers; bar is implicit)
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc='lower left')

    # 5) hide any empty subplots
    for empty_idx in range(n_ratings, nrows * ncols):
        row, col = divmod(empty_idx, ncols)
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()

    return metrics



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

def cronbach_alpha_with_plot(df: pd.DataFrame):
    # 1) Flatten the index and build a unique ItemID
    df_flat = df.reset_index()
    df_flat["ItemID"] = (
        df_flat["Category"].astype(str)
        + "_"
        + df_flat["Pose"].astype(str)
    )

    # 2) Compute alpha for each scale
    scales = ["Kawaii", "Expressive", "Warmth"]
    alphas = {}
    for scale in scales:
        sub = df_flat[df_flat["Rating"] == scale]
        wide = sub.pivot(index="RaterID", columns="ItemID", values="Score")
        alpha, _ = pg.cronbach_alpha(data=wide)
        alphas[scale] = alpha

    # 3) Plot the results
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(scales))
    vals = [alphas[s] for s in scales]

    bars = ax.bar(x, vals, width=0.6, edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cronbach’s α")
    ax.set_title("Scale Reliability")

    # threshold line at 0.70
    ax.axhline(0.70, color='gray', linestyle='--', label='α = 0.70')

    # annotate bars with exact values
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_similar_poses(df: pd.DataFrame,
                       rating: str,
                       threshold: float = 0.90,
                       top_n: int = 5) -> pd.DataFrame:
    """
    Identifies and plots pose–pose correlations for a given rating scale.
    - Resets any MultiIndex and removes duplicate columns.
    - Validates that `rating` exists in df["Rating"].
    - Pivots to a wide table with MultiIndex cols (Category, Pose).
    - Computes pose×pose correlations and stacks into long form:
        Category1, Pose1, Category2, Pose2, Correlation
    - Keeps pairs with corr ≥ threshold, else returns top_n by correlation.
    - Builds Pair labels as 'Category1 (Pose1) & Category2 (Pose2)'.
    - Plots a bar chart of these correlations.
    """
    # 1) Flatten any MultiIndex & drop duplicate columns
    df_flat = df.copy()
    if isinstance(df_flat.index, pd.MultiIndex):
        df_flat = df_flat.reset_index(allow_duplicates=True)
    df_flat = df_flat.loc[:, ~df_flat.columns.duplicated()]

    # 2) Validate rating
    available = df_flat["Rating"].unique()
    print("Available scales:", available)
    if rating not in available:
        raise ValueError(f"Rating '{rating}' not found. Choose from {available}")

    # 3) Subset to the specified scale
    sub = df_flat[df_flat["Rating"] == rating]

    # 4) Pivot to wide with MultiIndex cols: (Category, Pose)
    wide = sub.pivot_table(
        index="RaterID",
        columns=["Category", "Pose"],
        values="Score",
        aggfunc="mean"
    )
    print(f"[DEBUG] wide shape = {wide.shape}, items = {len(wide.columns)}")

    # 5) Compute pose×pose correlation matrix
    corr = wide.corr()
    corr.index.names = ["Category1", "Pose1"]
    corr.columns.names = ["Category2", "Pose2"]

    # 6) Stack into long form
    corr_pairs = corr.stack().reset_index(names=["Correlation"])

    # 7) Keep each unordered pair once
    mask_pairs = [
        (c1, p1) < (c2, p2)
        for c1, p1, c2, p2 in zip(
            corr_pairs["Category1"], corr_pairs["Pose1"],
            corr_pairs["Category2"], corr_pairs["Pose2"]
        )
    ]
    corr_pairs = corr_pairs[mask_pairs]

    # 8) Filter by threshold or grab top_n
    sim = corr_pairs[corr_pairs["Correlation"] >= threshold]
    if sim.empty:
        print(f"No pairs ≥ {threshold}; returning top {top_n}.")
        sim = corr_pairs.nlargest(top_n, "Correlation")
    else:
        sim = sim.sort_values("Correlation", ascending=False)

    # 9) Prepare labels: "Category (Pose) & Category (Pose)"
    sim["Pair"] = sim["Category1"] + " (" + sim["Pose1"] + ")" + " & " + \
                  sim["Category2"] + " (" + sim["Pose2"] + ")"

    # 10) Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sim["Pair"], sim["Correlation"])
    ax.axhline(threshold, linestyle="--", color="gray",
               label=f"threshold = {threshold}")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Pose–Pose Correlations ({rating})")
    ax.set_xticklabels(sim["Pair"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return sim


if __name__ == "__main__":
    df = load_df_from_csv("all_ratings.csv")
    # gaussian_scores(df)
    # metrics_per_category(df)
    # plot_correlation_matrix(df)
    # cronbach_alpha_with_plot(df)

    # plot_similar_poses(df, "Kawaii", threshold=0.9)

