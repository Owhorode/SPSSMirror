from typing import List, Dict
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column
from spssmirror.models._results import DescriptiveResult, FrequencyTableResult, CrossTabResult


class DescriptiveEngine:
    """
    Summary statistics, frequency tables, and crosstabs. This directly
    closes the gap surfaced when replicating a real thesis workflow in
    testing: v1 had no frequency-table capability at all.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def summary(self, column: str) -> DescriptiveResult:
        series = self._engine.get_column(column)
        validate_numeric_column(series, column)

        n_original = len(series)
        clean = series.dropna().astype(float)
        n = len(clean)
        n_missing = n_original - n

        if n == 0:
            raise ValueError(f"Column '{column}' has no valid numeric values after dropping nulls.")

        values = clean.to_numpy()
        mean = float(np.mean(values))
        median = float(np.median(values))
        mode_series = clean.mode()
        mode = float(mode_series.iloc[0]) if not mode_series.empty else None
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        skewness = float(scipy_stats.skew(values)) if n > 2 and std > 0 else None
        kurtosis = float(scipy_stats.kurtosis(values)) if n > 3 and std > 0 else None

        return DescriptiveResult(
            column=column, n=n, n_missing=n_missing, mean=mean, median=median, mode=mode,
            std=std, variance=variance, minimum=minimum, maximum=maximum,
            range_=maximum - minimum, q1=q1, q3=q3, iqr=q3 - q1,
            skewness=skewness, kurtosis=kurtosis, values=values.tolist(),
        )

    def summary_multiple(self, columns: List[str]) -> Dict[str, DescriptiveResult]:
        return {c: self.summary(c) for c in columns}

    def frequency_table(self, column: str, sort_by: str = "category") -> FrequencyTableResult:
        """
        sort_by: 'category' (default, sorted by value/label) or 'frequency'
        (descending count) — SPSS supports both orderings.
        """
        series = self._engine.get_column(column)
        n_missing = int(series.isna().sum())
        clean = series.dropna()
        n = len(clean)

        if n == 0:
            raise ValueError(f"Column '{column}' has no valid values after dropping nulls.")

        counts = clean.value_counts()
        if sort_by == "category":
            counts = counts.sort_index()
        elif sort_by == "frequency":
            counts = counts.sort_values(ascending=False)
        else:
            raise ValueError(f"sort_by must be 'category' or 'frequency', got '{sort_by}'")

        categories = [str(c) for c in counts.index.tolist()]
        frequencies = [int(f) for f in counts.tolist()]
        percentages = [f / n * 100 for f in frequencies]
        cumulative = np.cumsum(percentages).tolist() if sort_by == "category" else None

        return FrequencyTableResult(
            column=column, categories=categories, frequencies=frequencies,
            percentages=percentages,
            cumulative_percentages=cumulative if cumulative is not None else percentages,
            n=n, n_missing=n_missing,
        )

    def crosstab(self, row_col: str, col_col: str, with_chi_square: bool = True) -> CrossTabResult:
        row_series = self._engine.get_column(row_col)
        col_series = self._engine.get_column(col_col)

        mask = ~(row_series.isna() | col_series.isna())
        row_clean = row_series[mask]
        col_clean = col_series[mask]
        n = len(row_clean)

        if n == 0:
            raise ValueError(f"No overlapping non-null rows between '{row_col}' and '{col_col}'.")

        table = pd.crosstab(row_clean, col_clean)
        row_categories = [str(x) for x in table.index.tolist()]
        col_categories = [str(x) for x in table.columns.tolist()]
        counts_arr = table.to_numpy()
        counts = counts_arr.tolist()

        row_totals = counts_arr.sum(axis=1)
        row_totals_safe = np.where(row_totals == 0, 1, row_totals)
        row_pct = (counts_arr / row_totals_safe[:, None] * 100).tolist()

        chi2, chi2_p, cramers_v = None, None, None
        if with_chi_square and table.shape[0] > 1 and table.shape[1] > 1:
            try:
                chi2_stat, p_val, _, _ = scipy_stats.chi2_contingency(counts_arr)
                chi2, chi2_p = float(chi2_stat), float(p_val)
                min_dim = min(table.shape[0], table.shape[1]) - 1
                cramers_v = float(np.sqrt(chi2_stat / (n * min_dim))) if min_dim > 0 and n > 0 else None
            except ValueError:
                # e.g. a row/column of all zeros — leave as None instead of crashing
                pass

        return CrossTabResult(
            row_var=row_col, col_var=col_col, row_categories=row_categories,
            col_categories=col_categories, counts=counts, row_percentages=row_pct,
            chi_square=chi2, chi_square_p=chi2_p, cramers_v=cramers_v, n=n,
        )
