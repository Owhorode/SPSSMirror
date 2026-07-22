import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import (
    validate_numeric_column, validate_grouping_column, validate_min_sample_size,
)
from spssmirror.models._results import StatTestResult, DataQuality


class FrequentistNonparametricEngine:
    """
    Distribution-free counterparts to the t-test/ANOVA family. Every method
    stores per-group raw values in `groups[...]['values']` so the shared
    plot dispatcher can render boxplots (medians matter here, not means).
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _quality(self, n_used: int) -> DataQuality:
        n_original = self._engine.shape()[0]
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def mann_whitney_u(self, value_col: str, group_col: str, group1, group2) -> StatTestResult:
        value_series = self._engine.get_column(value_col)
        group_series = self._engine.get_column(group_col)
        validate_numeric_column(value_series, value_col)
        validate_grouping_column(group_series, group_col, min_groups=2)

        mask = ~(value_series.isna() | group_series.isna())
        v = value_series[mask].to_numpy(dtype=float)
        g = group_series[mask].to_numpy()
        v1, v2 = v[g == group1], v[g == group2]

        if len(v1) < 1 or len(v2) < 1:
            raise ValueError(f"Both groups need at least 1 observation. Got {len(v1)} and {len(v2)}.")

        u_stat, p_value = scipy_stats.mannwhitneyu(v1, v2, alternative="two-sided")
        n1, n2 = len(v1), len(v2)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else 0.0

        return StatTestResult(
            statistic=float(u_stat), p_value=float(p_value), effect_size=float(rank_biserial),
            effect_size_name="Rank-biserial correlation", n=n1 + n2,
            data_quality=self._quality(n1 + n2), test_name="Mann-Whitney U Test",
            groups={
                str(group1): {"median": float(np.median(v1)), "n": n1, "values": v1.tolist()},
                str(group2): {"median": float(np.median(v2)), "n": n2, "values": v2.tolist()},
            },
        )

    def wilcoxon_signed_rank(self, col1: str, col2: str) -> StatTestResult:
        s1, s2 = self._engine.get_column(col1), self._engine.get_column(col2)
        validate_numeric_column(s1, col1)
        validate_numeric_column(s2, col2)

        mask = ~(s1.isna() | s2.isna())
        a, b = s1[mask].to_numpy(dtype=float), s2[mask].to_numpy(dtype=float)
        diffs = a - b
        if np.all(diffs == 0):
            raise ValueError(
                f"Wilcoxon signed-rank test requires at least one non-zero difference "
                f"between '{col1}' and '{col2}'."
            )

        stat, p_value = scipy_stats.wilcoxon(a, b)
        n = len(a)
        z_approx = scipy_stats.norm.isf(p_value / 2)
        r_effect = float(z_approx / np.sqrt(n)) if n > 0 else 0.0

        return StatTestResult(
            statistic=float(stat), p_value=float(p_value), effect_size=r_effect,
            effect_size_name="r (Z / √N)", n=n, data_quality=self._quality(n),
            test_name="Wilcoxon Signed-Rank Test",
            groups={
                col1: {"median": float(np.median(a)), "n": n, "values": a.tolist()},
                col2: {"median": float(np.median(b)), "n": n, "values": b.tolist()},
            },
        )

    def kruskal_wallis(self, value_col: str, group_col: str) -> StatTestResult:
        value_series = self._engine.get_column(value_col)
        group_series = self._engine.get_column(group_col)
        validate_numeric_column(value_series, value_col)
        validate_grouping_column(group_series, group_col, min_groups=2)

        df_clean = pd.DataFrame({"v": value_series, "g": group_series}).dropna()
        grouped = list(df_clean.groupby("g"))
        labels = [str(g) for g, _ in grouped]
        arrays = [sub["v"].to_numpy(dtype=float) for _, sub in grouped]

        if any(len(a) < 1 for a in arrays):
            raise ValueError("Each group needs at least 1 observation for Kruskal-Wallis.")

        h_stat, p_value = scipy_stats.kruskal(*arrays)
        n = len(df_clean)
        k = len(arrays)
        epsilon_sq = (h_stat - k + 1) / (n - k) if (n - k) > 0 else None

        groups_dict = {
            labels[i]: {"median": float(np.median(arrays[i])), "n": len(arrays[i]), "values": arrays[i].tolist()}
            for i in range(k)
        }

        return StatTestResult(
            statistic=float(h_stat), p_value=float(p_value),
            effect_size=float(epsilon_sq) if epsilon_sq is not None else None,
            effect_size_name="Epsilon-squared", df=float(k - 1), n=n,
            data_quality=self._quality(n), test_name="Kruskal-Wallis H Test",
            groups=groups_dict,
        )

    def friedman_test(self, subject_col: str, condition_col: str, value_col: str) -> StatTestResult:
        df_clean = self._engine.to_dataframe()[[subject_col, condition_col, value_col]].dropna()
        pivot = df_clean.pivot(index=subject_col, columns=condition_col, values=value_col)

        if pivot.isna().any().any():
            raise ValueError(
                f"Friedman test requires a balanced design: every subject needs a value "
                f"for every level of '{condition_col}'. Some combinations are missing."
            )

        arrays = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
        stat, p_value = scipy_stats.friedmanchisquare(*arrays)
        n_subjects, k = pivot.shape[0], pivot.shape[1]
        kendalls_w = stat / (n_subjects * (k - 1)) if n_subjects * (k - 1) > 0 else None

        groups_dict = {
            str(col): {"median": float(np.median(pivot[col])), "n": n_subjects, "values": pivot[col].tolist()}
            for col in pivot.columns
        }

        return StatTestResult(
            statistic=float(stat), p_value=float(p_value),
            effect_size=float(kendalls_w) if kendalls_w is not None else None,
            effect_size_name="Kendall's W", df=float(k - 1), n=n_subjects * k,
            data_quality=self._quality(n_subjects * k), test_name="Friedman Test",
            groups=groups_dict,
        )
