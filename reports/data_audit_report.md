# Data Audit Report

This report provides a brief overview of the synthetic price and sentiment dataset used for the market risk detection project.

## Raw Data

Two raw data sources were used to construct the analytical dataset:

1. **Synthetic Price Series**: A simulated daily price series was generated for 365 days using a simple random walk with drift. Fields include:
   - `Date`: Calendar date.
   - `Close`: Simulated closing price.
   - `Volume`: Simulated trading volume.

2. **Synthetic Sentiment Scores**: Sentiment was proxied with a constant zero value for each day, since real news headline data was not available due to API restrictions.

## Processed Data

The analytical dataset (`analytical_data.csv`) contains the following columns:

| Column | Type | Description |
|-------|------|-------------|
| `Date` | String | Calendar date. |
| `Close` | Float | Simulated closing price. |
| `Log_Return` | Float | Daily log return: `ln(Close_t / Close_{t-1})`. |
| `Rolling_Volatility` | Float | Rolling standard deviation of log returns over a 5-day window. |
| `Rolling_Return_Mean` | Float | Rolling mean of log returns over a 5-day window. |
| `Rolling_Return_Std` | Float | Rolling standard deviation of log returns over a 5-day window (duplicate of `Rolling_Volatility`). |
| `Risk_Flag` | Integer (0/1) | Target variable indicating a high-risk day (1) if `Rolling_Volatility` exceeds its 90th percentile. |

## Diagnostics

- **Missing Values**: No missing values were present in the synthetic dataset after feature engineering.
- **Variable Distributions**: The distribution of log returns was approximately normal with slight skew; rolling volatility and mean exhibited low variance given the simulated nature of the data.

## Limitations

- **Synthetic Data**: Due to API rate limits and licensing restrictions, the dataset is entirely synthetic. It does not reflect real market conditions or sentiment dynamics.
- **Feature Simplification**: Only simple technical indicators were derived. Real-world models would benefit from additional features (e.g. macroeconomic indicators, alternative sentiment scores, volume-based metrics).

