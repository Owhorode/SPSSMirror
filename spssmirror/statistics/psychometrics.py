from typing import List, Optional
import numpy as np
from scipy import stats as scipy_stats
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_min_sample_size
from spssmirror.models._results import (
    PsychometricResult, DataQuality, ItemStat, ItemAnalysisResult, FactorAnalysisResult,
)


def _joint_clean_matrix(engine: DataEngine, items: List[str]) -> np.ndarray:
    for c in items:
        validate_numeric_column(engine.get_column(c), c)
    cols = [engine.get_column(c).to_numpy(dtype=float) for c in items]
    stacked = np.column_stack(cols)
    valid_mask = ~np.isnan(stacked).any(axis=1)
    return stacked[valid_mask]


def _cronbach_alpha_from_matrix(X: np.ndarray) -> float:
    k = X.shape[1]
    if k < 2:
        raise ValueError("Cronbach's Alpha requires at least 2 items.")
    cov = np.cov(X, rowvar=False, ddof=1)
    total_var = np.sum(cov)
    if total_var == 0:
        return 0.0
    return float((k / (k - 1)) * (1 - np.trace(cov) / total_var))


def _varimax(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Standard Kaiser varimax rotation. Returns rotated loadings; if there's
    only one factor, rotation is a no-op (nothing to rotate against)."""
    p, k = loadings.shape
    if k < 2:
        return loadings.copy()
    R = np.eye(k)
    d = 0.0
    L = loadings.copy()
    for _ in range(max_iter):
        Lambda = L @ R
        u, s, vt = np.linalg.svd(
            L.T @ (Lambda ** 3 - (1.0 / p) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if d != 0 and d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


class PsychometricsEngine:
    """
    Reliability, item analysis, and exploratory factor analysis. Factor
    extraction (used by both EFA and McDonald's omega) is implemented
    directly with numpy eigen-decomposition of the correlation matrix
    rather than depending on the third-party `factor_analyzer` package's
    `FactorAnalyzer.fit()`, which is currently broken against recent
    scikit-learn releases (calls a removed `check_array` argument). The
    package's `calculate_kmo`/`calculate_bartlett_sphericity` helpers don't
    hit that broken code path, so those are still used internally.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _quality(self, n_original: int, n_used: int) -> DataQuality:
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def cronbach_alpha(self, items: List[str]) -> PsychometricResult:
        if len(items) < 2:
            raise ValueError("Cronbach's Alpha requires at least 2 items.")
        n_original = self._engine.shape()[0]
        X = _joint_clean_matrix(self._engine, items)
        validate_min_sample_size(X.shape[0], 3, "Cronbach's Alpha")

        alpha = _cronbach_alpha_from_matrix(X)

        return PsychometricResult(
            statistic=alpha, n_items=len(items), n_respondents=X.shape[0], threshold=0.70,
            metric_name="Cronbach's Alpha", data_quality=self._quality(n_original, X.shape[0]),
        )

    def mcdonald_omega(self, items: List[str]) -> PsychometricResult:
        if len(items) < 2:
            raise ValueError("McDonald's Omega requires at least 2 items.")
        n_original = self._engine.shape()[0]
        X = _joint_clean_matrix(self._engine, items)
        validate_min_sample_size(X.shape[0], 3, "McDonald's Omega")

        loadings, _, _, _ = self._extract_factors(X, n_factors=1, rotation="none")
        lam = loadings[:, 0]
        sum_lam = np.sum(lam)
        sum_uniq = np.sum(1 - lam ** 2)
        denom = sum_lam ** 2 + sum_uniq
        omega = float((sum_lam ** 2) / denom) if denom > 0 else 0.0

        return PsychometricResult(
            statistic=omega, n_items=len(items), n_respondents=X.shape[0], threshold=0.70,
            metric_name="McDonald's Omega", data_quality=self._quality(n_original, X.shape[0]),
            details={"factor_loadings": lam.tolist()},
        )

    def split_half(self, items: List[str]) -> PsychometricResult:
        if len(items) < 4:
            raise ValueError("Split-half reliability needs at least 4 items to form two meaningful halves.")
        n_original = self._engine.shape()[0]
        X = _joint_clean_matrix(self._engine, items)
        validate_min_sample_size(X.shape[0], 3, "Split-half reliability")

        odd_idx = list(range(0, len(items), 2))
        even_idx = list(range(1, len(items), 2))
        half1 = X[:, odd_idx].sum(axis=1)
        half2 = X[:, even_idx].sum(axis=1)
        r, _ = scipy_stats.pearsonr(half1, half2)
        spearman_brown = float((2 * r) / (1 + r)) if (1 + r) != 0 else 0.0

        return PsychometricResult(
            statistic=spearman_brown, n_items=len(items), n_respondents=X.shape[0], threshold=0.70,
            metric_name="Split-Half Reliability (Spearman-Brown corrected)",
            data_quality=self._quality(n_original, X.shape[0]),
            details={"raw_half_correlation": float(r), "odd_items": [items[i] for i in odd_idx],
                     "even_items": [items[i] for i in even_idx]},
        )

    def kmo(self, items: List[str]) -> PsychometricResult:
        try:
            from factor_analyzer.factor_analyzer import calculate_kmo
        except ImportError:
            raise ImportError(
                "kmo() requires the 'factor_analyzer' package (used only for its "
                "standalone calculate_kmo function, not its FactorAnalyzer class -- "
                "see module docstring). Install with: pip install factor_analyzer"
            )

        n_original = self._engine.shape()[0]
        X = _joint_clean_matrix(self._engine, items)
        validate_min_sample_size(X.shape[0], len(items) + 1, "KMO test")

        try:
            kmo_per_var, kmo_total = calculate_kmo(X)
        except Exception as e:
            raise ValueError(f"KMO calculation failed: {e}") from e

        return PsychometricResult(
            statistic=float(kmo_total), n_items=len(items), n_respondents=X.shape[0], threshold=0.60,
            metric_name="Kaiser-Meyer-Olkin (KMO)", data_quality=self._quality(n_original, X.shape[0]),
            details={"per_item_msa": {items[i]: float(kmo_per_var[i]) for i in range(len(items))}},
        )

    def bartlett_sphericity(self, items: List[str]) -> PsychometricResult:
        try:
            from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
        except ImportError:
            raise ImportError(
                "bartlett_sphericity() requires the 'factor_analyzer' package (used "
                "only for its standalone calculate_bartlett_sphericity function). "
                "Install with: pip install factor_analyzer"
            )

        n_original = self._engine.shape()[0]
        X = _joint_clean_matrix(self._engine, items)
        validate_min_sample_size(X.shape[0], len(items) + 1, "Bartlett's test of sphericity")

        try:
            chi2, p_value = calculate_bartlett_sphericity(X)
        except Exception as e:
            raise ValueError(f"Bartlett's test failed: {e}") from e

        return PsychometricResult(
            statistic=float(chi2), n_items=len(items), n_respondents=X.shape[0],
            metric_name="Bartlett's Test of Sphericity", data_quality=self._quality(n_original, X.shape[0]),
            details={"p_value": float(p_value)},
        )

    def item_analysis(self, items: List[str]) -> ItemAnalysisResult:
        if len(items) < 2:
            raise ValueError("Item analysis requires at least 2 items.")
        n_original = self._engine.shape()[0]
        X = _joint_clean_matrix(self._engine, items)
        n = X.shape[0]
        overall_alpha = _cronbach_alpha_from_matrix(X)

        item_stats = []
        for i, name in enumerate(items):
            item_col = X[:, i]
            rest = np.delete(X, i, axis=1)
            rest_total = rest.sum(axis=1)
            if np.std(item_col) > 0 and np.std(rest_total) > 0:
                r_it, _ = scipy_stats.pearsonr(item_col, rest_total)
            else:
                r_it = 0.0
            alpha_wo = _cronbach_alpha_from_matrix(rest) if rest.shape[1] >= 2 else float("nan")

            item_stats.append(ItemStat(
                item=name, mean=float(np.mean(item_col)), std=float(np.std(item_col, ddof=1)),
                item_total_corr=float(r_it), alpha_if_deleted=float(alpha_wo),
            ))

        return ItemAnalysisResult(
            items=item_stats, overall_alpha=overall_alpha, n=n, n_items=len(items),
        )

    def efa(self, items: List[str], n_factors: Optional[int] = None, rotation: str = "varimax") -> FactorAnalysisResult:
        if len(items) < 3:
            raise ValueError("Factor analysis requires at least 3 items.")
        if rotation not in ("varimax", "none"):
            raise ValueError(f"rotation must be 'varimax' or 'none'. Got '{rotation}'.")

        X = _joint_clean_matrix(self._engine, items)
        validate_min_sample_size(X.shape[0], len(items) + 2, "Exploratory factor analysis")

        loadings, eigenvalues, communalities, k_used = self._extract_factors(X, n_factors=n_factors, rotation=rotation)

        variance_explained = (eigenvalues[:k_used] / len(items)).tolist()
        cumulative = np.cumsum(variance_explained).tolist()

        return FactorAnalysisResult(
            items=items, n_factors=k_used, loadings=loadings.tolist(),
            eigenvalues=eigenvalues.tolist(), variance_explained=variance_explained,
            cumulative_variance=cumulative, communalities=communalities.tolist(),
            rotation=rotation, n=X.shape[0],
        )

    @staticmethod
    def _extract_factors(X: np.ndarray, n_factors: Optional[int], rotation: str):
        """Principal-axis-style extraction via eigen-decomposition of the
        correlation matrix. Returns (loadings, eigenvalues_desc, communalities, k_used)."""
        k_items = X.shape[1]
        corr = np.corrcoef(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(corr)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if n_factors is None:
            k_used = int(np.sum(eigvals > 1.0))
            k_used = max(k_used, 1)
        else:
            k_used = n_factors
        k_used = min(k_used, k_items)

        pos_eigvals = np.clip(eigvals[:k_used], a_min=0, a_max=None)
        loadings = eigvecs[:, :k_used] * np.sqrt(pos_eigvals)

        if rotation == "varimax" and k_used >= 2:
            loadings = _varimax(loadings)

        communalities = np.sum(loadings ** 2, axis=1)
        return loadings, eigvals, communalities, k_used
