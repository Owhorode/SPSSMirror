from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from spssmirror._constants import LIKERT_SCALES, FUZZY_MATCH_THRESHOLD


class DataCleaner:
    def __init__(self):
        self.anomalies: Dict[str, List[int]] = {}

    def fuzzy_map_likert(self, series: pd.Series, scale_name: str = "agree_5",
                          threshold: float = FUZZY_MATCH_THRESHOLD) -> pd.Series:
        if scale_name not in LIKERT_SCALES:
            raise ValueError(f"Unknown scale '{scale_name}'. Available: {list(LIKERT_SCALES.keys())}")

        scale_dict = LIKERT_SCALES[scale_name]
        canonical_keys = list(scale_dict.keys())

        def map_value(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip().lower()
            best_match, best_score = None, 0.0
            for canonical in canonical_keys:
                score = fuzz.token_sort_ratio(val_str, canonical) / 100.0
                if score > best_score and score >= threshold:
                    best_score, best_match = score, canonical
            return scale_dict.get(best_match, np.nan) if best_match else np.nan

        return series.map(map_value)

    def auto_detect_dtype(self, series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "float"
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            n_unique = series.nunique(dropna=True)
            if n_unique <= 20 and len(series) >= 10:
                return "categorical"
            return "string"
        return "unknown"

    def isolate_infinite(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            finite = np.isfinite(df[col]) | df[col].isna()
            if not finite.all():
                self.anomalies[col] = df.index[~finite].tolist()
            mask &= finite
        return df[mask]

    def impute_missing_numeric(self, series: pd.Series, method: str = "mean") -> pd.Series:
        if series.isna().sum() == 0:
            return series
        if method == "mean":
            return series.fillna(series.mean())
        if method == "median":
            return series.fillna(series.median())
        if method == "forward_fill":
            return series.ffill()
        raise ValueError(f"Unknown imputation method: {method}")
