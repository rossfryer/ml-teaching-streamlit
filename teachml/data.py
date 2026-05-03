"""Synthetic datasets for supervised learning demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def linear_regression_1d(
    n_samples: int,
    slope: float,
    intercept: float,
    noise_std: float,
    *,
    seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate (x, y) where y ≈ slope * x + intercept + Gaussian noise.

    Returns a DataFrame with feature column "x" and the target vector y.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n_samples)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = slope * x + intercept + noise
    df = pd.DataFrame({"x": x})
    return df, y
