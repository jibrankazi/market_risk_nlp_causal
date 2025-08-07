"""
Data ingestion utilities for the market risk detection project.

This module provides functions for downloading financial time‑series data and
computing simple sentiment scores from text. The objective of these
functions is to centralize all data retrieval and basic preprocessing
logic so that notebooks and downstream scripts remain clean and focused
on analysis.

The functions defined here deliberately avoid any proprietary or
commercial APIs. Stock price data is fetched using the `yfinance`
package, which provides a convenient wrapper around Yahoo! Finance
without requiring an API key. Sentiment scores are derived using
`textblob`, a lightweight NLP library that offers a polarity score
between ‑1 (negative) and +1 (positive) for a given piece of text.

Note: In a production environment you would likely replace the simple
sentiment example with a more sophisticated model (e.g. FinBERT or
another domain‑specific transformer) and source real news articles via
licensed feeds. Here we keep things self‑contained and reproducible.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf
from textblob import TextBlob

def download_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical price data for a given ticker.

    Parameters
    ----------
    ticker : str
        The ticker symbol to download (e.g. "SPY" or "^GSPC").
    start_date : str
        Start date in ISO format (YYYY‑MM‑DD).
    end_date : str
        End date in ISO format (YYYY‑MM‑DD).
    interval : str, optional
        Sampling interval (e.g. "1d" for daily data). See yfinance docs.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date containing open, high, low, close,
        adjusted close and volume columns.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    # Ensure the index is a proper DatetimeIndex for further processing
    data.index = pd.to_datetime(data.index)
    # Remove rows with all NaN values
    return data.dropna(how="all")


def compute_sentiment_scores(texts: Iterable[str]) -> List[float]:
    """Compute sentiment polarity scores for a collection of texts.

    Parameters
    ----------
    texts : Iterable[str]
        An iterable of text strings (e.g. news headlines or tweets).

    Returns
    -------
    List[float]
        A list of polarity scores in the range [‑1, 1], where positive
        values indicate positive sentiment and negative values indicate
        negative sentiment.
    """
    scores: List[float] = []
    for text in texts:
        blob = TextBlob(text)
        scores.append(blob.sentiment.polarity)
    return scores


def fetch_sample_news() -> List[str]:
    """Return a small list of sample financial news headlines.

    This helper function provides a minimal set of text samples to
    illustrate sentiment analysis. In a real project you would replace
    this with code that pulls the latest news from a proper data source.
    """
    return [
        "Stocks rally as investors shrug off economic slowdown fears",
        "Central bank signals cautious approach amid inflation concerns",
        "Tech giant reports record profits beating analyst expectations",
        "Market volatility spikes after unexpected geopolitical tensions",
        "Energy sector dips on weak demand forecasts",
    ]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to disk in CSV format.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : Path
        The file path where the CSV should be written. The parent
        directory is created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def save_series(series: Iterable, path: Path) -> None:
    """Persist a series of values to disk in CSV format.

    Parameters
    ----------
    series : Iterable
        A sequence of values to save (e.g. sentiment scores).
    path : Path
        Destination file path.
    """
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for item in series:
            writer.writerow([item])


if __name__ == "__main__":
    # Example usage when executed as a script. Downloads price data for
    # SPY over the last year and computes sentiment for sample headlines.
    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    prices = download_price_data("SPY", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    news = fetch_sample_news()
    sentiments = compute_sentiment_scores(news)
    # Save to disk
    save_dataframe(prices, Path("data/raw/spy_prices.csv"))
    save_series(sentiments, Path("data/raw/sample_sentiments.csv"))