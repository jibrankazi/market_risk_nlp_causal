"""
Feature engineering utilities for the market risk detection project.

This module contains functions to transform raw price and sentiment
data into a structured analytical dataset suitable for machine
learning. Functions include calculation of daily returns, rolling
volatility metrics and merging of exogenous features such as
sentiment scores.

The engineered dataset can then be used to train models that
classify periods of elevated market risk or regress future price
movements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def calculate_returns(data: pd.DataFrame, price_col: str = "Adj Close") -> pd.Series:
    """Compute log returns for a price series.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing a price column.
    price_col : str, optional
        Name of the column containing prices. Defaults to "Adj Close".

    Returns
    -------
    pd.Series
        Series of log returns.
    """
    prices = data[price_col]
    returns = np.log(prices / prices.shift(1))
    return returns


def calculate_rolling_volatility(
    returns: pd.Series, window: int = 21
) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns).

    Parameters
    ----------
    returns : pd.Series
        Series of returns.
    window : int, optional
        Rolling window length in days. Defaults to 21 (~one trading month).

    Returns
    -------
    pd.Series
        Rolling standard deviation of returns.
    """
    return returns.rolling(window).std()


def create_analytical_dataset(
    price_data: pd.DataFrame,
    sentiment_scores: Optional[pd.Series] = None,
    risk_threshold: float = -0.02,
) -> pd.DataFrame:
    """Combine price and sentiment data into a single analytical DataFrame.

    The resulting DataFrame includes log returns, rolling volatility
    and optional sentiment features. A binary target column
    `Risk_Flag` is created indicating whether the next day's return
    falls below the specified risk threshold (i.e. large negative
    movement).

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing at least an "Adj Close" column and
        indexed by date.
    sentiment_scores : pd.Series, optional
        Series indexed by date containing sentiment scores. If
        provided, the sentiment will be forward‑filled to align with
        price dates.
    risk_threshold : float, optional
        Return threshold below which the following day is considered a
        risk event. Defaults to -0.02 (‑2%).

    Returns
    -------
    pd.DataFrame
        Analytical dataset with engineered features and target.
    """
    df = price_data.copy().sort_index()
    df["Log_Return"] = calculate_returns(df)
    df["Rolling_Vol"] = calculate_rolling_volatility(df["Log_Return"])
    # Shift returns to align with features: today's return influences tomorrow's target
    df["Next_Return"] = df["Log_Return"].shift(-1)
    # Binary target: 1 if next day's return is below threshold, else 0
    df["Risk_Flag"] = (df["Next_Return"] < risk_threshold).astype(int)
    # Include sentiment if provided
    if sentiment_scores is not None:
        # Ensure sentiment is a Series with datetime index
        sent = sentiment_scores.copy()
        sent.index = pd.to_datetime(sent.index)
        # Reindex to price dates and forward fill
        df["Sentiment"] = sent.reindex(df.index, method="ffill")
    # Drop rows with NaN due to shifting/rolling at the start/end
    df = df.dropna()
    return df


def save_analytical_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save the analytical DataFrame to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
