from typing import Dict, Any
import numpy as np
import pandas as pd


def validate_numeric_column(series: pd.Series, col_name: str, allow_null: bool = True) -> Dict[str, Any]:
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{col_name}' must be numeric. Got dtype {series.dtype}")
    null_count = int(series.isna().sum())
    if not allow_null and null_count > 0:
        raise ValueError(f"Column '{col_name}' contains {null_count} null values.")
    return {"valid": True, "null_count": null_count, "dtype": str(series.dtype)}


def validate_categorical_column(series: pd.Series, col_name: str) -> Dict[str, Any]:
    n_unique = series.nunique(dropna=True)
    return {"valid": True, "n_unique": int(n_unique), "dtype": str(series.dtype)}


def validate_grouping_column(series: pd.Series, col_name: str, min_groups: int = 2) -> Dict[str, Any]:
    n_unique = series.nunique(dropna=True)
    if n_unique < min_groups:
        raise ValueError(
            f"Column '{col_name}' has {n_unique} unique value(s), need at least {min_groups}."
        )
    return {"valid": True, "n_groups": int(n_unique), "group_labels": series.dropna().unique().tolist()}


def validate_min_sample_size(n: int, minimum: int, context: str = "this test") -> None:
    if n < minimum:
        raise ValueError(f"{context} requires at least {minimum} observations. Got {n}.")


def validate_has_variance(values: np.ndarray, col_name: str, context: str = "this test") -> None:
    """Parametric tests (t-tests, ANOVA) divide by sample variance/pooled
    variance internally. When a variable is exactly constant, that
    denominator is zero (or, worse, a tiny nonzero floating-point noise
    value from an internal design-matrix computation) -- which produces
    either a silent NaN result or, more dangerously, a spurious
    "significant" p-value computed from numerical noise rather than any
    real signal. Both are worse than refusing to run the test."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size > 0 and np.ptp(finite_values) == 0:
        raise ValueError(
            f"Column '{col_name}' is constant (every value equals "
            f"{finite_values[0]:g}) — {context} requires variability in the "
            f"outcome and cannot produce a meaningful result on constant data."
        )


def joint_valid_mask(*columns: pd.Series) -> np.ndarray:
    """Row-wise mask that is True only where ALL given columns are non-null.
    Prevents the row-misalignment bug found in earlier prototypes where each
    column was cleaned independently."""
    mask = np.ones(len(columns[0]), dtype=bool)
    for col in columns:
        mask &= ~pd.isna(col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col))
    return mask
