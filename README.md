# Real-Time Market Risk Detection with NLP & Causal Inference

## Project Overview
This project aims to build a real-time risk detection system for financial markets using natural language processing (NLP) and causal inference. It combines quantitative market data with unstructured text data (e.g., financial news) to identify emerging risks and potential market impacts.

## Motivation
Market volatility can be influenced by news, social media sentiment, and macroeconomic announcements. Detecting risk signals early helps investors and regulators respond proactively. This project explores how AI can synthesize textual sentiment and price movements to generate timely risk alerts.

## Repository Structure
- `data/` – storage for raw and processed datasets.
  - `raw/` – unmodified downloads of market prices and text data.
  - `processed/` – cleaned and engineered datasets ready for modelling.
- `notebooks/` – Jupyter notebooks for data analysis, feature engineering, and model development.
- `src/` – source code modules (data ingestion, NLP processing, causal inference models).
- `models/` – trained model files and weights.
- `reports/` – generated reports summarizing analyses and model performance.
- `visualizations/` – charts and plots generated during analysis.
- `.gitignore` – specifies untracked files and directories.
- `requirements.txt` – Python dependencies.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/jibrankazi/market_risk_nlp_causal.git
   cd market_risk_nlp_causal
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the notebooks in `notebooks/` to download data, perform feature engineering, and train the models.

## Data Sources
We plan to use:
- Historical stock price data via [Yahoo! Finance](https://finance.yahoo.com/) (accessible with the `yfinance` Python library).
- Financial sentiment text datasets such as the [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news).
- Additional macroeconomic indicators from public APIs where available.

## Methodology
The workflow will include:
1. **Data ingestion**: Download price time series and acquire text data.
2. **NLP sentiment analysis**: Use transformer-based models or lexicon-based approaches to assign sentiment scores to news articles.
3. **Feature engineering**: Combine price-derived indicators (volatility, returns) with aggregated sentiment features.
4. **Causal inference**: Apply methods such as Granger causality tests or causal impact modelling to understand directional influences between sentiment and market movements.
5. **Real-time pipeline**: Explore a prototype pipeline that ingests streaming data and produces risk signals.

## Results and Impact
Results will include model performance metrics (e.g., accuracy, precision) and visualisations such as correlation heatmaps, sentiment trends, and time-series plots of risk scores versus market movements. The goal is to demonstrate that combining NLP with causal inference can provide meaningful risk insights.

## Future Work
Potential enhancements include:
- Incorporating more data sources (e.g., social media via accessible APIs, macroeconomic indicators).
- Experimenting with advanced causal discovery algorithms.
- Deploying the model in a real-time environment with streaming ingestion and alerting.

## License
This project is licensed under the MIT License.
