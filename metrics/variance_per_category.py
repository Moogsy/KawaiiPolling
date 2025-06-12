import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def symmetric_log_transform(x):
    return np.sign(x - 1) * np.log10(np.abs(x - 1) + 1)

def compute_variance_per_category(df):
    """
    Compute rating variance (std dev) per pose grouped by category and scale.
    
    Parameters:
        df (pd.DataFrame): with columns ['Category', 'Pose', 'Rating', 'Score']
    
    Returns:
        pd.DataFrame: average std deviation per category and rating scale.
    """
    # Calculate std dev per pose within each category and rating scale
    pose_variance = df.groupby(['Category', 'Pose', 'Rating'])['Score'].std().reset_index(name='StdDev')
    
    # Average the std devs per category and rating
    category_variance = pose_variance.groupby(['Category', 'Rating'])['StdDev'].mean().reset_index()
    
    return category_variance


def plot_variability_heatmap_symmetric_log(df):

    category_variance_df = compute_variance_per_category(df)
    
    pivot = category_variance_df.pivot(index='Category', columns='Rating', values='StdDev')

    # Apply symmetric log transform
    sym_log_data = pivot.applymap(symmetric_log_transform)

    plt.figure(figsize=(10,6))
    sns.heatmap(
        sym_log_data,
        annot=pivot.round(2),  # annotate original std dev
        fmt="",
        cmap="coolwarm",
        center=0  # center at zero for 1.0 original value
    )

    plt.title("Average Rating Variability (Std Dev) per Category and Scale (Symmetric log scale)")
    plt.ylabel("Category")
    plt.xlabel("Rating Scale")
    plt.show()

