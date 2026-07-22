from typing import List, Optional
import numpy as np
import statsmodels.formula.api as smf
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_min_sample_size, validate_has_variance
from spssmirror.models._results import (
    MixedModelResult, RegressionCoefficient, RandomEffectVarianceComponent,
    GroupRandomEffect, DataQuality,
)


class MixedModelsEngine:
    """
    Linear mixed-effects models (random intercepts and/or random slopes)
    wrapping statsmodels MixedLM. statsmodels is used internally only —
    every method returns a SPSSMirror result model.

    Design notes:
    - Random effects are specified via `groups` (the clustering variable)
      and optional `re_formula` (e.g. '~time' for a random slope on time;
      omit for random-intercept-only, the common default).
    - AIC/BIC are only well-defined under maximum likelihood (ML), not
      restricted maximum likelihood (REML) — statsmodels itself returns
      NaN for them under REML. Rather than fabricate values, this engine
      reports them as None when REML is used (the default, since REML
      gives less biased variance component estimates for a single model).
      Use method='ml' explicitly if you need AIC/BIC for model comparison.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _quality(self, n_original: int, n_used: int) -> DataQuality:
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def linear_mixed_model(self, formula: str, group_col: str, re_formula: Optional[str] = None,
                            method: str = "reml") -> MixedModelResult:
        if method not in ("reml", "ml"):
            raise ValueError(f"method must be 'reml' or 'ml'. Got '{method}'.")
        if "~" not in formula:
            raise ValueError(f"Invalid formula '{formula}'. Expected e.g. 'score ~ time + treatment'.")

        y_name = formula.split("~")[0].strip()
        involved_cols = [y_name, group_col]
        df_full = self._engine.to_dataframe()
        if y_name not in df_full.columns:
            raise ValueError(f"Dependent variable '{y_name}' not found in data.")
        if group_col not in df_full.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in data.")

        n_original = len(df_full)
        df_clean = df_full.dropna(subset=[c for c in df_full.columns if c in formula or c == group_col])
        n_used = len(df_clean)
        validate_min_sample_size(n_used, 5, "Linear mixed model")

        n_groups = df_clean[group_col].nunique()
        if n_groups < 2:
            raise ValueError(
                f"Mixed models require at least 2 groups in '{group_col}'. Got {n_groups}."
            )
        validate_has_variance(df_clean[y_name].to_numpy(dtype=float), y_name, "Linear mixed model")

        try:
            model = smf.mixedlm(formula, df_clean, groups=df_clean[group_col], re_formula=re_formula)
            fit = model.fit(reml=(method == "reml"))
        except Exception as e:
            raise ValueError(
                f"Mixed model failed to converge: {e}. This can happen with too few "
                f"groups, too little within-group variation, or a mis-specified "
                f"random-effects structure."
            ) from e

        fe_names = list(fit.fe_params.index)
        fixed_effects = []
        for name in fe_names:
            se = float(fit.bse_fe.get(name, np.nan))
            p = float(fit.pvalues.get(name, np.nan)) if name in fit.pvalues.index else None
            z = float(fit.tvalues.get(name, np.nan)) if hasattr(fit, "tvalues") and name in fit.tvalues.index else None
            ci_lo, ci_hi = None, None
            if hasattr(fit, "conf_int"):
                try:
                    conf = fit.conf_int()
                    if name in conf.index:
                        ci_lo, ci_hi = float(conf.loc[name, 0]), float(conf.loc[name, 1])
                except Exception:
                    pass
            fixed_effects.append(RegressionCoefficient(
                term=name, b=float(fit.fe_params[name]), std_error=se,
                z_value=z, p_value=p, ci_lower=ci_lo, ci_upper=ci_hi,
            ))

        cov_re = fit.cov_re
        variance_components = [
            RandomEffectVarianceComponent(name=str(idx), variance=float(cov_re.loc[idx, idx]))
            for idx in cov_re.index
        ]
        residual_var = float(fit.scale)
        variance_components.append(RandomEffectVarianceComponent(name="Residual", variance=residual_var))

        icc = None
        if len(cov_re.index) >= 1:
            intercept_var = float(cov_re.iloc[0, 0])
            denom = intercept_var + residual_var
            icc = intercept_var / denom if denom > 0 else None

        group_effects = []
        for group_key, effects_series in fit.random_effects.items():
            group_effects.append(GroupRandomEffect(
                group=str(group_key), effects={str(k): float(v) for k, v in effects_series.items()},
            ))

        aic = float(fit.aic) if method == "ml" and fit.aic is not None and not np.isnan(fit.aic) else None
        bic = float(fit.bic) if method == "ml" and fit.bic is not None and not np.isnan(fit.bic) else None
        llf = float(fit.llf) if fit.llf is not None else None

        return MixedModelResult(
            fixed_effects=fixed_effects, variance_components=variance_components,
            group_effects=group_effects, icc=icc, group_col=group_col, n_groups=int(n_groups),
            n=n_used, log_likelihood=llf, aic=aic, bic=bic,
            estimation_method=method.upper(), y_name=y_name,
            converged=bool(fit.converged) if hasattr(fit, "converged") else True,
            data_quality=self._quality(n_original, n_used),
        )
