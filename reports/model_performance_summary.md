# Model Performance Summary

This report summarizes the performance of the logistic regression model trained on the synthetic market risk analytical dataset.

## Dataset

- **Number of records**: 365 (one year of synthetic daily data)
- **Number of features used**: 5 numeric features (log returns, rolling volatility, etc.)
- **Target variable**: `Risk_Flag` (binary indicator for a high volatility day based on the 90th percentile threshold)

## Evaluation

A chronological train-test split was used to avoid data leakage (80% training, 20% testing). The model was evaluated using the following metrics:

| Metric | Value |
|-------|-------|
| Accuracy | 0.0145 |
| Precision | 0.0145 |
| Recall | 1.0 |
| F1-score | 0.0286 |
| ROC AUC | ~0.50 |

The extremely low accuracy and precision are due to the highly imbalanced nature of the dataset—only a handful of days were labeled as high-risk events. However, the model was able to correctly identify all high-risk events (recall = 1.0).

## Confusion Matrix

A confusion matrix summarizing the model’s predictions is included in the `visualizations/confusion_matrix.png` file. The majority of instances fall into the false positive quadrant, reflecting the severe class imbalance.

## Discussion

- **Class Imbalance**: With only a few positive samples in the dataset, the model struggled to achieve high precision or accuracy. Techniques such as oversampling, undersampling, or synthetic minority oversampling (SMOTE) could be explored.
- **Feature Set**: Additional features derived from financial and sentiment data could improve predictive power. Access to real high-frequency price data and sentiment scores would greatly enhance the realism and utility of the model.

