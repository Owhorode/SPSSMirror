from typing import List, Optional
import numpy as np
import pandas as pd
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_min_sample_size
from spssmirror.models._results import (
    SurvivalPoint, KaplanMeierResult, LogRankResult, CoxCoefficient,
    CoxPHResult, ParametricSurvivalResult,
)

_PARAMETRIC_DISTRIBUTIONS = ("weibull", "exponential", "lognormal", "loglogistic")


class SurvivalEngine:
    """
    Time-to-event analysis: Kaplan-Meier curves, log-rank tests, Cox
    proportional hazards, and parametric survival models. Built on
    `lifelines` internally only — every method returns a SPSSMirror
    result model, never a raw lifelines object.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _get_duration_event(self, duration_col: str, event_col: str):
        from spssmirror.preprocessing._validators import joint_valid_mask

        d_series = self._engine.get_column(duration_col)
        e_series = self._engine.get_column(event_col)
        validate_numeric_column(d_series, duration_col)
        validate_numeric_column(e_series, event_col)

        mask = joint_valid_mask(d_series, e_series)
        duration = d_series.to_numpy(dtype=float)[mask]
        event = e_series.to_numpy(dtype=float)[mask]

        if np.any(duration < 0):
            raise ValueError(f"Duration column '{duration_col}' cannot contain negative values.")
        uniques = set(np.unique(event).tolist())
        if not uniques.issubset({0.0, 1.0}):
            raise ValueError(
                f"Event column '{event_col}' must be binary (1=event occurred, 0=censored). "
                f"Got values: {sorted(uniques)}"
            )
        return duration, event

    def kaplan_meier(self, duration_col: str, event_col: str, group_col: Optional[str] = None,
                      confidence: float = 0.95) -> List[KaplanMeierResult]:
        from lifelines import KaplanMeierFitter

        if group_col is None:
            duration, event = self._get_duration_event(duration_col, event_col)
            validate_min_sample_size(len(duration), 3, "Kaplan-Meier estimation")
            return [self._fit_km(duration, event, None, confidence)]

        d_series = self._engine.get_column(duration_col)
        e_series = self._engine.get_column(event_col)
        g_series = self._engine.get_column(group_col)
        validate_numeric_column(d_series, duration_col)
        validate_numeric_column(e_series, event_col)

        mask = ~(d_series.isna() | e_series.isna() | g_series.isna())
        df = pd.DataFrame({
            "duration": d_series[mask].to_numpy(dtype=float),
            "event": e_series[mask].to_numpy(dtype=float),
            "group": g_series[mask].to_numpy(),
        })

        if df["group"].nunique() < 2:
            raise ValueError(f"group_col '{group_col}' must have at least 2 distinct groups.")

        results = []
        for group_val, sub in df.groupby("group"):
            validate_min_sample_size(len(sub), 3, f"Kaplan-Meier estimation for group '{group_val}'")
            results.append(self._fit_km(
                sub["duration"].to_numpy(), sub["event"].to_numpy(), str(group_val), confidence,
            ))
        return results

    @staticmethod
    def _fit_km(duration: np.ndarray, event: np.ndarray, label: Optional[str], confidence: float) -> KaplanMeierResult:
        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        try:
            kmf.fit(duration, event_observed=event, label=label or "curve", alpha=1 - confidence)
        except Exception as e:
            raise ValueError(f"Kaplan-Meier fit failed: {e}") from e

        sf = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_
        at_risk_events = kmf.event_table

        points = []
        for t in sf.index:
            n_at_risk = int(at_risk_events.loc[t, "at_risk"]) if t in at_risk_events.index else 0
            n_events = int(at_risk_events.loc[t, "observed"]) if t in at_risk_events.index else 0
            ci_lo = float(ci.iloc[:, 0].loc[t]) if t in ci.index else float(sf.loc[t])
            ci_hi = float(ci.iloc[:, 1].loc[t]) if t in ci.index else float(sf.loc[t])
            points.append(SurvivalPoint(
                time=float(t), survival_prob=float(sf.loc[t]), ci_lower=ci_lo, ci_upper=ci_hi,
                n_at_risk=n_at_risk, n_events=n_events,
            ))

        median = kmf.median_survival_time_
        median_clean = float(median) if median is not None and np.isfinite(median) else None

        return KaplanMeierResult(
            group=label, curve=points, median_survival=median_clean,
            n=len(duration), n_events=int(np.sum(event)),
        )

    def logrank_test(self, duration_col: str, event_col: str, group_col: str) -> LogRankResult:
        from lifelines.statistics import multivariate_logrank_test

        d_series = self._engine.get_column(duration_col)
        e_series = self._engine.get_column(event_col)
        g_series = self._engine.get_column(group_col)
        validate_numeric_column(d_series, duration_col)
        validate_numeric_column(e_series, event_col)

        mask = ~(d_series.isna() | e_series.isna() | g_series.isna())
        duration = d_series[mask].to_numpy(dtype=float)
        event = e_series[mask].to_numpy(dtype=float)
        group = g_series[mask].to_numpy()

        groups_unique = sorted(set(str(g) for g in group))
        if len(groups_unique) < 2:
            raise ValueError(f"group_col '{group_col}' must have at least 2 distinct groups for a log-rank test.")

        try:
            result = multivariate_logrank_test(duration, group, event)
        except Exception as e:
            raise ValueError(f"Log-rank test failed: {e}") from e

        return LogRankResult(
            statistic=float(result.test_statistic), p_value=float(result.p_value),
            groups_compared=groups_unique, n=len(duration),
        )

    def cox_ph(self, duration_col: str, event_col: str, covariates: List[str]) -> CoxPHResult:
        from lifelines import CoxPHFitter

        if not covariates:
            raise ValueError("cox_ph requires at least one covariate.")

        cols = [duration_col, event_col] + covariates
        df = self._engine.to_dataframe()[cols].dropna()
        n = len(df)
        validate_min_sample_size(n, len(covariates) + 5, "Cox proportional hazards model")

        uniques = set(df[event_col].unique().tolist())
        if not uniques.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Event column '{event_col}' must be binary (1=event, 0=censored).")

        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col=duration_col, event_col=event_col)
        except Exception as e:
            raise ValueError(
                f"Cox PH model failed to fit: {e}. This can happen with perfect "
                f"separation, too few events relative to covariates, or severe "
                f"collinearity among covariates."
            ) from e

        summary = cph.summary
        coefficients = []
        for cov in summary.index:
            row = summary.loc[cov]
            coefficients.append(CoxCoefficient(
                covariate=cov, coef=float(row["coef"]), hazard_ratio=float(row["exp(coef)"]),
                std_error=float(row["se(coef)"]), z_value=float(row["z"]), p_value=float(row["p"]),
                ci_lower=float(row["coef lower 95%"]), ci_upper=float(row["coef upper 95%"]),
                hr_ci_lower=float(row["exp(coef) lower 95%"]), hr_ci_upper=float(row["exp(coef) upper 95%"]),
            ))

        return CoxPHResult(
            coefficients=coefficients, concordance=float(cph.concordance_index_),
            log_likelihood=float(cph.log_likelihood_), aic=float(cph.AIC_partial_),
            n=n, n_events=int(df[event_col].sum()), baseline_hazard_available=True,
        )

    def parametric_survival(self, duration_col: str, event_col: str, covariates: List[str],
                             distribution: str = "weibull") -> ParametricSurvivalResult:
        if distribution not in _PARAMETRIC_DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {_PARAMETRIC_DISTRIBUTIONS}. Got '{distribution}'.")
        if not covariates:
            raise ValueError("parametric_survival requires at least one covariate.")

        from lifelines import WeibullAFTFitter, ExponentialFitter, LogNormalAFTFitter, LogLogisticAFTFitter

        fitter_map = {
            "weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter,
            "loglogistic": LogLogisticAFTFitter,
        }

        cols = [duration_col, event_col] + covariates
        df = self._engine.to_dataframe()[cols].dropna()
        n = len(df)
        validate_min_sample_size(n, len(covariates) + 5, "Parametric survival model")

        if distribution == "exponential":
            df_exp = df[[duration_col, event_col]].copy()
            model = ExponentialFitter()
            try:
                model.fit(df_exp[duration_col], event_observed=df_exp[event_col])
            except Exception as e:
                raise ValueError(f"Exponential survival model failed to fit: {e}") from e
            coefficients = [CoxCoefficient(
                covariate="lambda_", coef=float(model.lambda_), hazard_ratio=float(np.exp(model.lambda_)),
                std_error=0.0, z_value=0.0, p_value=1.0, ci_lower=0.0, ci_upper=0.0,
                hr_ci_lower=0.0, hr_ci_upper=0.0,
            )]
            return ParametricSurvivalResult(
                distribution=distribution, coefficients=coefficients,
                log_likelihood=float(model.log_likelihood_), aic=float(model.AIC_),
                n=n, n_events=int(df[event_col].sum()),
            )

        try:
            model = fitter_map[distribution]()
            model.fit(df, duration_col=duration_col, event_col=event_col)
        except Exception as e:
            raise ValueError(f"{distribution.capitalize()} survival model failed to fit: {e}") from e

        summary = model.summary
        coefficients = []
        for idx in summary.index:
            row = summary.loc[idx]
            param_name = idx[0] if isinstance(idx, tuple) else str(idx)
            cov_name = idx[1] if isinstance(idx, tuple) else str(idx)
            coefficients.append(CoxCoefficient(
                covariate=f"{param_name}::{cov_name}", coef=float(row["coef"]),
                hazard_ratio=float(np.exp(row["coef"])), std_error=float(row["se(coef)"]),
                z_value=float(row["z"]), p_value=float(row["p"]),
                ci_lower=float(row["coef lower 95%"]), ci_upper=float(row["coef upper 95%"]),
                hr_ci_lower=float(np.exp(row["coef lower 95%"])), hr_ci_upper=float(np.exp(row["coef upper 95%"])),
            ))

        return ParametricSurvivalResult(
            distribution=distribution, coefficients=coefficients,
            log_likelihood=float(model.log_likelihood_), aic=float(model.AIC_),
            n=n, n_events=int(df[event_col].sum()),
        )
