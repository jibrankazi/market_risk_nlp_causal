"""
Visualization utilities for the market risk detection project.

This module encapsulates common plotting functions used throughout the
analysis. By centralizing plotting logic, we ensure consistent
styling and make it easy to generate figures for both exploratory
analysis and final presentations. All functions save plots directly
to disk to facilitate later upload into the GitHub repository and
portfolio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def plot_price_series(
    data: pd.DataFrame, price_col: str = "Adj Close", title: str = "Price Series", path: Path | None = None
) -> None:
    """Plot the time series of an asset's price.

    Parameters
    ----------
    data : pd.DataFrame
        Price data indexed by date.
    price_col : str, optional
        Column name for the price. Defaults to "Adj Close".
    title : str, optional
        Figure title.
    path : Path, optional
        If provided, the figure will be saved to this path.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data[price_col], label=price_col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    plt.close()


def plot_return_distribution(
    returns: pd.Series, title: str = "Return Distribution", path: Path | None = None
) -> None:
    """Plot a histogram of returns.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(returns.dropna(), kde=True, bins=50)
    plt.title(title)
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    path: Path | None = None,
) -> None:
    """Plot a confusion matrix heatmap.
    """
    if class_names is None:
        class_names = ["No Risk", "Risk"]
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    plt.close()
