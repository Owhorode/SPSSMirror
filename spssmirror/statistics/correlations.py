from typing import List, Tuple
import numpy as np
from scipy import stats as scipy_stats
import statsmodels.api as sm
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_min_sample_size
from spssmirror.models._results import CorrelationResult, CorrelationMatrixResult, DataQuality


class CorrelationEngine:
    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _quality(self, n_used: int) -> DataQuality:
        n_original = self._engine.shape()[0]
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def _paired_clean(self, col1: str, col2: str) -> Tuple[np.ndarray, np.ndarray]:
        s1, s2 = self._engine.get_column(col1), self._engine.get_column(col2)
        validate_numeric_column(s1, col1)
        validate_numeric_column(s2, col2)
        mask = ~(s1.isna() | s2.isna())
        return s1[mask].to_numpy(dtype=float), s2[mask].to_numpy(dtype=float)

    @staticmethod
    def _fisher_ci(r: float, n: int, confidence: float) -> Tuple[float, float]:
        if abs(r) >= 0.999 or n <= 3:
            return float(r), float(r)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = scipy_stats.norm.ppf((1 + confidence) / 2)
        return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))

    def pearson(self, col1: str, col2: str, confidence: float = 0.95) -> CorrelationResult:
        x, y = self._paired_clean(col1, col2)
        validate_min_sample_size(len(x), 3, "Pearson correlation")
        r, p_value = scipy_stats.pearsonr(x, y)
        ci_lo, ci_hi = self._fisher_ci(r, len(x), confidence)

        return CorrelationResult(
            coefficient=float(r), p_value=float(p_value), n=len(x), method="Pearson",
            ci_lower=ci_lo, ci_upper=ci_hi, data_quality=self._quality(len(x)),
            x_name=col1, y_name=col2, x_values=x.tolist(), y_values=y.tolist(),
        )

    def spearman(self, col1: str, col2: str, confidence: float = 0.95) -> CorrelationResult:
        x, y = self._paired_clean(col1, col2)
        validate_min_sample_size(len(x), 3, "Spearman correlation")
        rho, p_value = scipy_stats.spearmanr(x, y)
        ci_lo, ci_hi = self._fisher_ci(rho, len(x), confidence)

        return CorrelationResult(
            coefficient=float(rho), p_value=float(p_value), n=len(x), method="Spearman",
            ci_lower=ci_lo, ci_upper=ci_hi, data_quality=self._quality(len(x)),
            x_name=col1, y_name=col2, x_values=x.tolist(), y_values=y.tolist(),
        )

    def kendall_tau(self, col1: str, col2: str) -> CorrelationResult:
        x, y = self._paired_clean(col1, col2)
        validate_min_sample_size(len(x), 3, "Kendall's Tau")
        tau, p_value = scipy_stats.kendalltau(x, y)

        return CorrelationResult(
            coefficient=float(tau), p_value=float(p_value), n=len(x), method="Kendall's Tau",
            data_quality=self._quality(len(x)), x_name=col1, y_name=col2,
            x_values=x.tolist(), y_values=y.tolist(),
        )

    def point_biserial(self, binary_col: str, continuous_col: str) -> CorrelationResult:
        s1 = self._engine.get_column(binary_col)
        s2 = self._engine.get_column(continuous_col)
        validate_numeric_column(s2, continuous_col)

        mask = ~(s1.isna() | s2.isna())
        b = s1[mask].to_numpy()
        c = s2[mask].to_numpy(dtype=float)
        uniques = set(np.unique(b).tolist())
        if not uniques.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"point_biserial requires '{binary_col}' to be binary (0/1). Got values: {sorted(uniques)}"
            )

        r, p_value = scipy_stats.pointbiserialr(b.astype(float), c)

        return CorrelationResult(
            coefficient=float(r), p_value=float(p_value), n=len(b), method="Point-Biserial",
            data_quality=self._quality(len(b)), x_name=binary_col, y_name=continuous_col,
            x_values=b.astype(float).tolist(), y_values=c.tolist(),
        )

    def partial(self, col1: str, col2: str, covariates: List[str]) -> CorrelationResult:
        if not covariates:
            raise ValueError("partial() requires at least one covariate. Use pearson() for zero-order correlation.")

        cols = [col1, col2] + covariates
        df_clean = self._engine.to_dataframe()[cols].dropna()
        n = len(df_clean)
        validate_min_sample_size(n, len(covariates) + 3, "Partial correlation")

        X_cov = sm.add_constant(df_clean[covariates].to_numpy(dtype=float))
        y1 = df_clean[col1].to_numpy(dtype=float)
        y2 = df_clean[col2].to_numpy(dtype=float)
        resid1 = y1 - sm.OLS(y1, X_cov).fit().fittedvalues
        resid2 = y2 - sm.OLS(y2, X_cov).fit().fittedvalues

        r, p_value = scipy_stats.pearsonr(resid1, resid2)

        return CorrelationResult(
            coefficient=float(r), p_value=float(p_value), n=n,
            method=f"Partial Pearson (controlling for {', '.join(covariates)})",
            data_quality=self._quality(n), x_name=col1, y_name=col2,
            x_values=resid1.tolist(), y_values=resid2.tolist(),
        )

    def correlation_matrix(self, columns: List[str], method: str = "pearson") -> CorrelationMatrixResult:
        if method not in ("pearson", "spearman", "kendall"):
            raise ValueError(f"method must be 'pearson', 'spearman', or 'kendall'. Got '{method}'.")
        if len(columns) < 2:
            raise ValueError("correlation_matrix requires at least 2 columns.")

        df_clean = self._engine.to_dataframe()[columns].dropna()
        n = len(df_clean)
        validate_min_sample_size(n, 3, "Correlation matrix")

        func = {
            "pearson": scipy_stats.pearsonr,
            "spearman": scipy_stats.spearmanr,
            "kendall": scipy_stats.kendalltau,
        }[method]

        k = len(columns)
        matrix = np.eye(k)
        p_matrix = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                r, p = func(df_clean[columns[i]].to_numpy(dtype=float), df_clean[columns[j]].to_numpy(dtype=float))
                matrix[i, j] = matrix[j, i] = r
                p_matrix[i, j] = p_matrix[j, i] = p

        return CorrelationMatrixResult(
            variables=columns, matrix=matrix.tolist(), p_matrix=p_matrix.tolist(),
            method=method.capitalize(), n=n,
        )
