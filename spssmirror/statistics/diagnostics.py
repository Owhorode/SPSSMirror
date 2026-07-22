from typing import List
import numpy as np
import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from spssmirror.core._engine import DataEngine
from spssmirror.core._formula import parse_formula
from spssmirror.preprocessing._validators import (
    validate_numeric_column, validate_grouping_column, validate_min_sample_size, validate_has_variance,
)
from spssmirror.models._results import (
    NormalityTestStat, NormalityResult, VIFItem, VIFResult,
    ResidualDiagnosticsResult, OutlierResult, StatTestResult, DataQuality,
)


class DiagnosticsEngine:
    """
    Assumption-checking tools: normality, homogeneity of variance,
    multicollinearity (VIF), residual/leverage/Cook's-distance diagnostics,
    and outlier detection. scipy/statsmodels are used internally only.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _quality(self, n_original: int, n_used: int) -> DataQuality:
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def normality_tests(self, column: str) -> NormalityResult:
        series = self._engine.get_column(column)
        validate_numeric_column(series, column)
        clean = series.dropna().to_numpy(dtype=float)
        validate_min_sample_size(len(clean), 3, "Normality tests")
        validate_has_variance(clean, column, "Normality tests")
        n = len(clean)
        mean, std = float(np.mean(clean)), float(np.std(clean, ddof=1))

        tests = []

        sw_stat, sw_p = scipy_stats.shapiro(clean)
        tests.append(NormalityTestStat(test_name="Shapiro-Wilk", statistic=float(sw_stat), p_value=float(sw_p)))

        if std > 0:
            ks_stat, ks_p = scipy_stats.kstest(clean, "norm", args=(mean, std))
            tests.append(NormalityTestStat(test_name="Kolmogorov-Smirnov", statistic=float(ks_stat), p_value=float(ks_p)))

        if n >= 8:
            dag_stat, dag_p = scipy_stats.normaltest(clean)
            tests.append(NormalityTestStat(test_name="D'Agostino K-squared", statistic=float(dag_stat), p_value=float(dag_p)))

        ad_result = scipy_stats.anderson(clean, dist="norm", method="interpolate")
        tests.append(NormalityTestStat(
            test_name="Anderson-Darling", statistic=float(ad_result.statistic),
            p_value=float(ad_result.pvalue),
        ))

        p_values = [t.p_value for t in tests if t.p_value is not None]
        is_normal = all(p > 0.05 for p in p_values)

        skew = float(scipy_stats.skew(clean)) if std > 0 else None
        kurt = float(scipy_stats.kurtosis(clean)) if std > 0 else None

        return NormalityResult(
            column=column, tests=tests, n=n, mean=mean, std=std,
            skewness=skew, kurtosis=kurt, is_normal=bool(is_normal), values=clean.tolist(),
        )

    def homogeneity_of_variance(self, value_col: str, group_col: str, method: str = "levene") -> StatTestResult:
        if method not in ("levene", "bartlett"):
            raise ValueError(f"method must be 'levene' or 'bartlett'. Got '{method}'.")

        value_series = self._engine.get_column(value_col)
        group_series = self._engine.get_column(group_col)
        validate_numeric_column(value_series, value_col)
        validate_grouping_column(group_series, group_col, min_groups=2)
        n_original = self._engine.shape()[0]

        mask = ~(value_series.isna() | group_series.isna())
        v = value_series[mask].to_numpy(dtype=float)
        g = group_series[mask].to_numpy()
        groups = [v[g == lvl] for lvl in np.unique(g)]
        groups = [gr for gr in groups if len(gr) > 0]

        if any(len(gr) < 2 for gr in groups):
            raise ValueError("Each group needs at least 2 observations for a homogeneity-of-variance test.")

        if method == "levene":
            stat, p_value = scipy_stats.levene(*groups, center="median")
            test_name = "Levene's Test"
        else:
            stat, p_value = scipy_stats.bartlett(*groups)
            test_name = "Bartlett's Test"

        n_used = sum(len(gr) for gr in groups)
        return StatTestResult(
            statistic=float(stat), p_value=float(p_value), n=n_used,
            df=float(len(groups) - 1), data_quality=self._quality(n_original, n_used), test_name=test_name,
        )

    def vif(self, columns: List[str]) -> VIFResult:
        if len(columns) < 2:
            raise ValueError("VIF requires at least 2 predictor columns.")
        for c in columns:
            validate_numeric_column(self._engine.get_column(c), c)

        df_clean = self._engine.to_dataframe()[columns].dropna()
        n = len(df_clean)
        validate_min_sample_size(n, len(columns) + 2, "VIF")

        X = sm.add_constant(df_clean.to_numpy(dtype=float))
        items = []
        for i, name in enumerate(columns):
            try:
                v = float(variance_inflation_factor(X, i + 1))  # +1 to skip the constant column
            except Exception:
                v = float("inf")
            tolerance = 1 / v if v > 0 else 0.0
            items.append(VIFItem(predictor=name, vif=v, tolerance=tolerance))

        return VIFResult(items=items, n=n)

    def residual_diagnostics(self, formula: str) -> ResidualDiagnosticsResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        try:
            fit = sm.OLS(parsed.y, parsed.X).fit()
        except Exception as e:
            raise ValueError(f"Could not fit model for residual diagnostics: {e}") from e

        influence = fit.get_influence()
        leverage = influence.hat_matrix_diag
        std_resid = influence.resid_studentized_internal
        cooks_d = influence.cooks_distance[0]

        n = parsed.n_obs
        k = parsed.X.shape[1]
        leverage_cutoff = 2 * k / n if n > 0 else 0
        cooks_cutoff = 4 / n if n > 0 else 0
        influential = [
            i for i in range(n)
            if leverage[i] > leverage_cutoff or cooks_d[i] > cooks_cutoff
        ]

        return ResidualDiagnosticsResult(
            fitted=fit.fittedvalues.tolist(), residuals=fit.resid.tolist(),
            standardized_residuals=std_resid.tolist(), leverage=leverage.tolist(),
            cooks_distance=cooks_d.tolist(), influential_indices=influential,
            n=n, k_predictors=k, y_name=parsed.y_name,
        )

    def outliers(self, column: str, method: str = "iqr", threshold: float = 1.5) -> OutlierResult:
        if method not in ("iqr", "zscore"):
            raise ValueError(f"method must be 'iqr' or 'zscore'. Got '{method}'.")

        series = self._engine.get_column(column)
        validate_numeric_column(series, column)
        clean = series.dropna().to_numpy(dtype=float)
        validate_min_sample_size(len(clean), 4, "Outlier detection")

        if method == "iqr":
            q1, q3 = np.percentile(clean, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        else:
            mean, std = np.mean(clean), np.std(clean, ddof=1)
            lower, upper = mean - threshold * std, mean + threshold * std

        outlier_mask = (clean < lower) | (clean > upper)
        outlier_idx = np.where(outlier_mask)[0].tolist()

        return OutlierResult(
            column=column, method=method, lower_bound=float(lower), upper_bound=float(upper),
            outlier_indices=outlier_idx, outlier_values=clean[outlier_mask].tolist(),
            n=len(clean), n_outliers=len(outlier_idx), values=clean.tolist(),
        )
