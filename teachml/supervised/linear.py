"""Ordinary least squares linear regression (teaching wrapper around sklearn)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class LinearRegressionResult:
    intercept: float
    coefficients: np.ndarray
    feature_names: list[str]
    model: LinearRegression

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def fit_linear_regression(X: pd.DataFrame, y: np.ndarray) -> LinearRegressionResult:
    model = LinearRegression()
    model.fit(X, y)
    coefs = np.asarray(model.coef_, dtype=float).ravel()
    return LinearRegressionResult(
        intercept=float(model.intercept_),
        coefficients=coefs,
        feature_names=list(X.columns),
        model=model,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
