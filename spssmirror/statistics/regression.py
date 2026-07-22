from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from spssmirror.core._engine import DataEngine
from spssmirror.core._formula import parse_formula
from spssmirror.preprocessing._validators import validate_has_variance
from spssmirror.models._results import RegressionResult, RegressionCoefficient, DataQuality


_GLM_FAMILIES = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
    "poisson": sm.families.Poisson,
    "gamma": sm.families.Gamma,
}


def _try_array(fit, attr: str) -> Optional[np.ndarray]:
    try:
        val = getattr(fit, attr)
        if val is None:
            return None
        if callable(val):
            val = val()
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0 or np.all(np.isnan(arr)):
            return None
        return arr
    except Exception:
        return None


def _extract_coefficients(fit, x_names: List[str], standardized_betas: Optional[np.ndarray] = None,
                           use_z: bool = False) -> List[RegressionCoefficient]:
    params = np.asarray(fit.params, dtype=float)
    bse = _try_array(fit, "bse")
    pvalues = _try_array(fit, "pvalues")
    tvalues = _try_array(fit, "tvalues")
    conf = _try_array(fit, "conf_int")

    coefs = []
    for i, name in enumerate(x_names):
        coefs.append(RegressionCoefficient(
            term=name,
            b=float(params[i]),
            std_error=float(bse[i]) if bse is not None else None,
            t_value=(None if use_z else (float(tvalues[i]) if tvalues is not None else None)),
            z_value=(float(tvalues[i]) if use_z and tvalues is not None else None),
            p_value=float(pvalues[i]) if pvalues is not None else None,
            ci_lower=float(conf[i][0]) if conf is not None else None,
            ci_upper=float(conf[i][1]) if conf is not None else None,
            beta=(float(standardized_betas[i])
                  if standardized_betas is not None and not np.isnan(standardized_betas[i])
                  else None),
        ))
    return coefs


def _standardized_betas(y: np.ndarray, X: np.ndarray, x_names: List[str], b: np.ndarray) -> np.ndarray:
    sd_y = np.std(y, ddof=1)
    betas = np.full(len(x_names), np.nan)
    if sd_y == 0:
        return betas
    for i, name in enumerate(x_names):
        if name == "Intercept":
            continue
        sd_x = np.std(X[:, i], ddof=1)
        betas[i] = b[i] * (sd_x / sd_y) if sd_x > 0 else 0.0
    return betas


def _manual_r2(y: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _validate_binary(y: np.ndarray, context: str) -> None:
    uniques = set(np.unique(y).tolist())
    if not uniques.issubset({0.0, 1.0}):
        raise ValueError(
            f"{context} requires a binary (0/1) outcome variable. "
            f"Got values: {sorted(uniques)}. Recode your outcome to 0/1 first."
        )


def _validate_nonnegative_counts(y: np.ndarray, context: str) -> None:
    if np.any(y < 0):
        raise ValueError(f"{context} requires non-negative count values in the outcome variable.")


class RegressionEngine:
    """
    Every method here returns a RegressionResult — statsmodels objects never
    leave this module. This closes the gap found in earlier testing where
    users had to drop into raw statsmodels to run regression at all.
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

    # ------------------------------------------------------------------
    # Linear (OLS)
    # ------------------------------------------------------------------
    def linear(self, formula: str) -> RegressionResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        validate_has_variance(parsed.y, parsed.y_name, "Linear regression")
        try:
            fit = sm.OLS(parsed.y, parsed.X).fit()
        except Exception as e:
            raise ValueError(f"Linear regression failed to converge: {e}") from e

        std_betas = _standardized_betas(parsed.y, parsed.X, parsed.x_names, fit.params)
        coefs = _extract_coefficients(fit, parsed.x_names, standardized_betas=std_betas)

        return RegressionResult(
            model_type="Linear Regression (OLS)",
            r_squared=float(fit.rsquared), adj_r_squared=float(fit.rsquared_adj),
            f_statistic=float(fit.fvalue) if fit.fvalue is not None else None,
            f_p_value=float(fit.f_pvalue) if fit.f_pvalue is not None else None,
            log_likelihood=float(fit.llf), aic=float(fit.aic), bic=float(fit.bic),
            n=parsed.n_obs, df_model=int(fit.df_model), df_residual=int(fit.df_resid),
            coefficients=coefs, residuals=fit.resid.tolist(), fitted_values=fit.fittedvalues.tolist(),
            y_name=parsed.y_name, data_quality=self._quality(parsed.n_obs),
        )

    def linear_columns(self, y_col: str, x_cols: List[str]) -> RegressionResult:
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        return self.linear(formula)

    # ------------------------------------------------------------------
    # Logistic
    # ------------------------------------------------------------------
    def logistic(self, formula: str) -> RegressionResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        _validate_binary(parsed.y, "Logistic regression")
        try:
            fit = sm.Logit(parsed.y, parsed.X).fit(disp=0)
        except Exception as e:
            raise ValueError(
                f"Logistic regression failed to converge (often caused by perfect "
                f"separation or too few observations per predictor): {e}"
            ) from e

        coefs = _extract_coefficients(fit, parsed.x_names, use_z=True)
        fitted_probs = fit.predict(parsed.X)

        return RegressionResult(
            model_type="Logistic Regression",
            pseudo_r_squared=float(fit.prsquared),
            log_likelihood=float(fit.llf), aic=float(fit.aic), bic=float(fit.bic),
            n=parsed.n_obs, df_model=int(fit.df_model), df_residual=int(fit.df_resid),
            coefficients=coefs, residuals=(parsed.y - fitted_probs).tolist(),
            fitted_values=fitted_probs.tolist(),
            y_name=parsed.y_name, data_quality=self._quality(parsed.n_obs),
        )

    # ------------------------------------------------------------------
    # Poisson (count regression)
    # ------------------------------------------------------------------
    def poisson(self, formula: str) -> RegressionResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        _validate_nonnegative_counts(parsed.y, "Poisson regression")
        try:
            fit = sm.Poisson(parsed.y, parsed.X).fit(disp=0)
        except Exception as e:
            raise ValueError(f"Poisson regression failed to converge: {e}") from e

        coefs = _extract_coefficients(fit, parsed.x_names, use_z=True)
        fitted_vals = fit.predict(parsed.X)

        return RegressionResult(
            model_type="Poisson Regression",
            pseudo_r_squared=float(fit.prsquared),
            log_likelihood=float(fit.llf), aic=float(fit.aic), bic=float(fit.bic),
            n=parsed.n_obs, df_model=int(fit.df_model), df_residual=int(fit.df_resid),
            coefficients=coefs, residuals=(parsed.y - fitted_vals).tolist(),
            fitted_values=fitted_vals.tolist(),
            y_name=parsed.y_name, data_quality=self._quality(parsed.n_obs),
        )

    # ------------------------------------------------------------------
    # Generalized Linear Model
    # ------------------------------------------------------------------
    def glm(self, formula: str, family: str = "gaussian") -> RegressionResult:
        if family not in _GLM_FAMILIES:
            raise ValueError(f"family must be one of {list(_GLM_FAMILIES.keys())}, got '{family}'")

        parsed = parse_formula(formula, self._engine.to_dataframe())
        if family == "binomial":
            _validate_binary(parsed.y, "Binomial GLM")
        if family in ("poisson", "gamma"):
            _validate_nonnegative_counts(parsed.y, f"{family.capitalize()} GLM")

        try:
            fit = sm.GLM(parsed.y, parsed.X, family=_GLM_FAMILIES[family]()).fit()
        except Exception as e:
            raise ValueError(f"GLM ({family}) failed to converge: {e}") from e

        coefs = _extract_coefficients(fit, parsed.x_names, use_z=True)
        fitted_vals = fit.fittedvalues
        pseudo_r2 = (1 - fit.deviance / fit.null_deviance) if fit.null_deviance not in (0, None) else None

        return RegressionResult(
            model_type=f"GLM ({family.capitalize()} family)",
            pseudo_r_squared=float(pseudo_r2) if pseudo_r2 is not None else None,
            log_likelihood=float(fit.llf) if fit.llf is not None else None,
            aic=float(fit.aic) if fit.aic is not None else None,
            bic=float(fit.bic_llf) if hasattr(fit, "bic_llf") and fit.bic_llf is not None else None,
            n=parsed.n_obs, df_model=int(fit.df_model), df_residual=int(fit.df_resid),
            coefficients=coefs, residuals=(parsed.y - fitted_vals).tolist(),
            fitted_values=fitted_vals.tolist(),
            y_name=parsed.y_name, data_quality=self._quality(parsed.n_obs),
        )

    # ------------------------------------------------------------------
    # Robust regression (Huber M-estimation) — resists outliers
    # ------------------------------------------------------------------
    def robust(self, formula: str) -> RegressionResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        validate_has_variance(parsed.y, parsed.y_name, "Robust regression")
        try:
            fit = sm.RLM(parsed.y, parsed.X, M=sm.robust.norms.HuberT()).fit()
        except Exception as e:
            raise ValueError(f"Robust regression failed to converge: {e}") from e

        coefs = _extract_coefficients(fit, parsed.x_names)
        fitted_vals = fit.fittedvalues
        r2 = _manual_r2(parsed.y, fitted_vals)

        return RegressionResult(
            model_type="Robust Regression (Huber M-estimation)",
            r_squared=r2,
            n=parsed.n_obs, df_model=len(parsed.x_names) - 1, df_residual=int(fit.df_resid),
            coefficients=coefs, residuals=fit.resid.tolist(), fitted_values=fitted_vals.tolist(),
            y_name=parsed.y_name, data_quality=self._quality(parsed.n_obs),
        )

    # ------------------------------------------------------------------
    # Regularized regression: ridge / lasso / elastic net
    # No classical SE/p-values — regularization invalidates them. Coeffs
    # come back with std_error/p_value/ci as None, honestly.
    # ------------------------------------------------------------------
    def _regularized(self, formula: str, alpha: float, l1_ratio: float, label: str) -> RegressionResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        validate_has_variance(parsed.y, parsed.y_name, f"{label} regression")
        try:
            fit = sm.OLS(parsed.y, parsed.X).fit_regularized(alpha=alpha, L1_wt=l1_ratio)
        except Exception as e:
            raise ValueError(f"{label} regression failed: {e}") from e

        params = np.asarray(fit.params, dtype=float)
        fitted_vals = parsed.X @ params
        r2 = _manual_r2(parsed.y, fitted_vals)

        coefs = [RegressionCoefficient(term=name, b=float(params[i]))
                 for i, name in enumerate(parsed.x_names)]

        return RegressionResult(
            model_type=f"{label} Regression (alpha={alpha}, l1_ratio={l1_ratio})",
            r_squared=r2,
            n=parsed.n_obs, df_model=len(parsed.x_names) - 1,
            df_residual=parsed.n_obs - len(parsed.x_names),
            coefficients=coefs, residuals=(parsed.y - fitted_vals).tolist(),
            fitted_values=fitted_vals.tolist(),
            y_name=parsed.y_name, data_quality=self._quality(parsed.n_obs),
        )

    def ridge(self, formula: str, alpha: float = 1.0) -> RegressionResult:
        return self._regularized(formula, alpha=alpha, l1_ratio=0.0, label="Ridge")

    def lasso(self, formula: str, alpha: float = 1.0) -> RegressionResult:
        return self._regularized(formula, alpha=alpha, l1_ratio=1.0, label="Lasso")

    def elastic_net(self, formula: str, alpha: float = 1.0, l1_ratio: float = 0.5) -> RegressionResult:
        return self._regularized(formula, alpha=alpha, l1_ratio=l1_ratio, label="Elastic Net")
