from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_grouping_column, validate_min_sample_size
from spssmirror.models._results import EffectSizeResult


def _interpret(value: float, small: float, medium: float, large: float) -> str:
    v = abs(value)
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def _bootstrap_ci(compute_fn, n: int, n_boot: int = 1000, confidence: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = compute_fn(idx)
    alpha = (1 - confidence) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


class EffectSizeEngine:
    """
    Standalone effect size calculators with confidence intervals and
    conventional (Cohen, 1988) small/medium/large interpretation labels.
    Complements the effect sizes already attached to test results — useful
    when you want an effect size independent of running the full test.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _groups(self, value_col: str, group_col: str, group1, group2):
        value_series = self._engine.get_column(value_col)
        group_series = self._engine.get_column(group_col)
        validate_numeric_column(value_series, value_col)
        validate_grouping_column(group_series, group_col, min_groups=2)
        mask = ~(value_series.isna() | group_series.isna())
        v = value_series[mask].to_numpy(dtype=float)
        g = group_series[mask].to_numpy()
        v1, v2 = v[g == group1], v[g == group2]
        validate_min_sample_size(len(v1), 2, f"Effect size for group '{group1}'")
        validate_min_sample_size(len(v2), 2, f"Effect size for group '{group2}'")
        return v1, v2

    def cohens_d(self, value_col: str, group_col: str, group1, group2, confidence: float = 0.95) -> EffectSizeResult:
        v1, v2 = self._groups(value_col, group_col, group1, group2)
        n1, n2 = len(v1), len(v2)
        pooled_sd = np.sqrt(((n1 - 1) * np.var(v1, ddof=1) + (n2 - 1) * np.var(v2, ddof=1)) / (n1 + n2 - 2))
        d = (np.mean(v1) - np.mean(v2)) / pooled_sd if pooled_sd > 0 else 0.0

        se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
        z = scipy_stats.norm.ppf((1 + confidence) / 2)
        ci_lo, ci_hi = d - z * se, d + z * se

        return EffectSizeResult(
            name="Cohen's d", value=float(d), ci_lower=float(ci_lo), ci_upper=float(ci_hi),
            interpretation=_interpret(d, 0.2, 0.5, 0.8), n=n1 + n2,
            details={"n1": n1, "n2": n2, "mean1": float(np.mean(v1)), "mean2": float(np.mean(v2))},
        )

    def hedges_g(self, value_col: str, group_col: str, group1, group2, confidence: float = 0.95) -> EffectSizeResult:
        v1, v2 = self._groups(value_col, group_col, group1, group2)
        n1, n2 = len(v1), len(v2)
        pooled_sd = np.sqrt(((n1 - 1) * np.var(v1, ddof=1) + (n2 - 1) * np.var(v2, ddof=1)) / (n1 + n2 - 2))
        d = (np.mean(v1) - np.mean(v2)) / pooled_sd if pooled_sd > 0 else 0.0

        df = n1 + n2 - 2
        correction = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
        g = d * correction

        se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
        z = scipy_stats.norm.ppf((1 + confidence) / 2)
        ci_lo, ci_hi = g - z * se, g + z * se

        return EffectSizeResult(
            name="Hedges' g", value=float(g), ci_lower=float(ci_lo), ci_upper=float(ci_hi),
            interpretation=_interpret(g, 0.2, 0.5, 0.8), n=n1 + n2,
            details={"n1": n1, "n2": n2, "correction_factor": float(correction)},
        )

    def glass_delta(self, value_col: str, group_col: str, group1, group2, control_group: str,
                     confidence: float = 0.95) -> EffectSizeResult:
        if control_group not in (group1, group2):
            raise ValueError(f"control_group must be either '{group1}' or '{group2}'.")
        v1, v2 = self._groups(value_col, group_col, group1, group2)
        control_vals = v1 if control_group == group1 else v2
        treatment_vals = v2 if control_group == group1 else v1
        control_sd = np.std(control_vals, ddof=1)

        delta = (np.mean(treatment_vals) - np.mean(control_vals)) / control_sd if control_sd > 0 else 0.0
        n1, n2 = len(v1), len(v2)
        se = np.sqrt((n1 + n2) / (n1 * n2) + delta ** 2 / (2 * (n2 - 1))) if n2 > 1 else float("nan")
        z = scipy_stats.norm.ppf((1 + confidence) / 2)
        ci_lo, ci_hi = delta - z * se, delta + z * se

        return EffectSizeResult(
            name="Glass's Delta", value=float(delta), ci_lower=float(ci_lo), ci_upper=float(ci_hi),
            interpretation=_interpret(delta, 0.2, 0.5, 0.8), n=n1 + n2,
            details={"control_group": control_group, "control_sd": float(control_sd)},
        )

    def eta_squared(self, value_col: str, group_col: str, confidence: float = 0.95,
                     n_boot: int = 1000) -> EffectSizeResult:
        df_clean = pd.DataFrame({
            "v": self._engine.get_column(value_col), "g": self._engine.get_column(group_col),
        }).dropna()
        n = len(df_clean)
        validate_min_sample_size(n, 3, "Eta-squared")

        def compute(idx):
            sub = df_clean.iloc[idx]
            grand_mean = sub["v"].mean()
            ss_total = ((sub["v"] - grand_mean) ** 2).sum()
            ss_between = sub.groupby("g")["v"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
            return ss_between / ss_total if ss_total > 0 else 0.0

        eta_sq = compute(np.arange(n))
        ci_lo, ci_hi = _bootstrap_ci(compute, n, n_boot=n_boot, confidence=confidence)

        return EffectSizeResult(
            name="Eta-squared", value=float(eta_sq), ci_lower=ci_lo, ci_upper=ci_hi,
            interpretation=_interpret(eta_sq, 0.01, 0.06, 0.14), n=n,
            details={"ci_method": f"bootstrap (n_boot={n_boot})"},
        )

    def omega_squared(self, value_col: str, group_col: str, confidence: float = 0.95,
                       n_boot: int = 1000) -> EffectSizeResult:
        df_clean = pd.DataFrame({
            "v": self._engine.get_column(value_col), "g": self._engine.get_column(group_col),
        }).dropna()
        n = len(df_clean)
        validate_min_sample_size(n, 3, "Omega-squared")
        k = df_clean["g"].nunique()

        def compute(idx):
            sub = df_clean.iloc[idx]
            grand_mean = sub["v"].mean()
            n_sub = len(sub)
            ss_total = ((sub["v"] - grand_mean) ** 2).sum()
            ss_between = sub.groupby("g")["v"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
            ss_within = ss_total - ss_between
            ms_within = ss_within / (n_sub - k) if (n_sub - k) > 0 else 0.0
            denom = ss_total + ms_within
            return (ss_between - (k - 1) * ms_within) / denom if denom > 0 else 0.0

        omega_sq = compute(np.arange(n))
        ci_lo, ci_hi = _bootstrap_ci(compute, n, n_boot=n_boot, confidence=confidence)

        return EffectSizeResult(
            name="Omega-squared", value=float(omega_sq), ci_lower=ci_lo, ci_upper=ci_hi,
            interpretation=_interpret(omega_sq, 0.01, 0.06, 0.14), n=n,
            details={"k_groups": int(k), "ci_method": f"bootstrap (n_boot={n_boot})"},
        )

    def cramers_v(self, col1: str, col2: str, confidence: float = 0.95, n_boot: int = 1000) -> EffectSizeResult:
        s1, s2 = self._engine.get_column(col1), self._engine.get_column(col2)
        mask = ~(s1.isna() | s2.isna())
        df_clean = pd.DataFrame({"a": s1[mask].reset_index(drop=True), "b": s2[mask].reset_index(drop=True)})
        n = len(df_clean)
        validate_min_sample_size(n, 5, "Cramer's V")

        def compute(idx):
            sub = df_clean.iloc[idx]
            table = pd.crosstab(sub["a"], sub["b"])
            if table.shape[0] < 2 or table.shape[1] < 2:
                return 0.0
            chi2, _, _, _ = scipy_stats.chi2_contingency(table.to_numpy())
            min_dim = min(table.shape) - 1
            return float(np.sqrt(chi2 / (len(sub) * min_dim))) if min_dim > 0 else 0.0

        v = compute(np.arange(n))
        ci_lo, ci_hi = _bootstrap_ci(compute, n, n_boot=n_boot, confidence=confidence)

        return EffectSizeResult(
            name="Cramer's V", value=float(v), ci_lower=ci_lo, ci_upper=ci_hi,
            interpretation=_interpret(v, 0.1, 0.3, 0.5), n=n,
            details={"ci_method": f"bootstrap (n_boot={n_boot})"},
        )

    def odds_ratio(self, col1: str, col2: str, confidence: float = 0.95) -> EffectSizeResult:
        s1, s2 = self._engine.get_column(col1), self._engine.get_column(col2)
        mask = ~(s1.isna() | s2.isna())
        table = pd.crosstab(s1[mask], s2[mask])
        if table.shape != (2, 2):
            raise ValueError(f"odds_ratio requires a 2x2 table. Got {table.shape[0]}x{table.shape[1]}.")

        a, b = table.iloc[0, 0], table.iloc[0, 1]
        c, d = table.iloc[1, 0], table.iloc[1, 1]
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5  # Haldane-Anscombe correction for zero cells

        odds_ratio = (a * d) / (b * c)
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = scipy_stats.norm.ppf((1 + confidence) / 2)
        ci_lo, ci_hi = np.exp(log_or - z * se_log_or), np.exp(log_or + z * se_log_or)

        return EffectSizeResult(
            name="Odds Ratio", value=float(odds_ratio), ci_lower=float(ci_lo), ci_upper=float(ci_hi),
            interpretation=_interpret(np.log(odds_ratio), np.log(1.5), np.log(2.5), np.log(4.0)),
            n=int(table.to_numpy().sum()), details={"table": table.to_numpy().tolist()},
        )
