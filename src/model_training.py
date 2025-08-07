"""
Model training utilities for the market risk detection project.

This module defines functions for splitting the analytical dataset,
training a simple classification model to predict market risk events
and evaluating model performance. We choose a logistic regression
classifier here for interpretability; however, the same pipeline can
be extended to more complex models (random forests, gradient boosting,
etc.) if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


@dataclass
class ModelResults:
    model: LogisticRegression
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray


def train_test_split_time(
    data: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a chronological split on the data.

    Parameters
    ----------
    data : pd.DataFrame
        Analytical dataset sorted by date.
    test_size : float, optional
        Fraction of the data to reserve for testing. Defaults to 0.2.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing subsets.
    """
    n = len(data)
    split_idx = int(n * (1 - test_size))
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    return train_df, test_df


def train_logistic_model(
    data: pd.DataFrame, feature_cols: list[str], target_col: str = "Risk_Flag"
) -> ModelResults:
    """Train a logistic regression classifier and compute metrics.

    Parameters
    ----------
    data : pd.DataFrame
        Analytical dataset containing features and target.
    feature_cols : list of str
        Names of the columns to use as model inputs.
    target_col : str, optional
        Name of the target column. Defaults to "Risk_Flag".

    Returns
    -------
    ModelResults
        Dataclass containing trained model and evaluation scores.
    """
    # Split data chronologically to avoid leakage
    train_df, test_df = train_test_split_time(data)
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    # Fit logistic regression with balanced class weights
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    conf = confusion_matrix(y_test, y_pred)
    return ModelResults(model, acc, prec, rec, f1, roc, conf)
