from typing import List, Optional, Tuple
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf, adfuller
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import validate_numeric_column, validate_min_sample_size
from spssmirror.models._results import (
    ARIMACoefficient, ForecastPoint, ARIMAResult, ExponentialSmoothingResult,
    GARCHResult, ACFResult, StationarityResult,
)


class TimeSeriesEngine:
    """
    ARIMA/SARIMA, exponential smoothing, GARCH, and diagnostic tools
    (ACF/PACF, stationarity). statsmodels/arch are used internally only —
    every method returns a SPSSMirror result model.

    Design note: auto-order selection (`auto_arima`) is implemented as a
    direct AIC grid search over statsmodels' own `ARIMA` class rather than
    depending on the third-party `pmdarima` package, which has a history
    of dependency-compatibility breaks similar to the ones already found
    in `factor_analyzer` (Phase 5) and `arviz`/`pymc` (Phase 9) during this
    build.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _get_series(self, column: str) -> np.ndarray:
        series = self._engine.get_column(column)
        validate_numeric_column(series, column)
        clean = series.dropna().to_numpy(dtype=float)
        validate_min_sample_size(len(clean), 10, "Time series analysis")
        return clean

    def _forecast_points(self, mean: np.ndarray, ci_lower: np.ndarray, ci_upper: np.ndarray) -> List[ForecastPoint]:
        return [
            ForecastPoint(step=i + 1, mean=float(mean[i]), ci_lower=float(ci_lower[i]), ci_upper=float(ci_upper[i]))
            for i in range(len(mean))
        ]

    def arima(self, column: str, order: Tuple[int, int, int] = (1, 0, 0),
              seasonal_order: Optional[Tuple[int, int, int, int]] = None,
              forecast_steps: int = 0) -> ARIMAResult:
        y = self._get_series(column)
        if any(o < 0 for o in order):
            raise ValueError(f"ARIMA order components must be non-negative. Got {order}.")

        try:
            kwargs = {"order": order}
            if seasonal_order is not None:
                kwargs["seasonal_order"] = seasonal_order
            model = ARIMA(y, **kwargs)
            fit = model.fit()
        except Exception as e:
            raise ValueError(f"ARIMA{order} failed to fit: {e}") from e

        coefficients = [
            ARIMACoefficient(
                term=name, value=float(fit.params[i]),
                std_error=float(fit.bse[i]) if fit.bse is not None else None,
                p_value=float(fit.pvalues[i]) if fit.pvalues is not None else None,
            )
            for i, name in enumerate(fit.param_names)
        ]

        forecast_points = []
        if forecast_steps > 0:
            fc = fit.get_forecast(steps=forecast_steps)
            ci_arr = np.asarray(fc.conf_int(alpha=0.05))
            forecast_points = self._forecast_points(
                np.asarray(fc.predicted_mean), ci_arr[:, 0], ci_arr[:, 1],
            )

        return ARIMAResult(
            order=list(order), seasonal_order=list(seasonal_order) if seasonal_order else None,
            coefficients=coefficients, aic=float(fit.aic), bic=float(fit.bic),
            log_likelihood=float(fit.llf), fitted_values=np.asarray(fit.fittedvalues).tolist(),
            residuals=np.asarray(fit.resid).tolist(), original_values=y.tolist(),
            forecast=forecast_points, n=len(y), y_name=column,
        )

    def auto_arima(self, column: str, max_p: int = 3, max_d: int = 2, max_q: int = 3,
                    forecast_steps: int = 0) -> ARIMAResult:
        """AIC grid search over (p,d,q) combinations up to the given maxima."""
        y = self._get_series(column)
        best_aic = np.inf
        best_order = None
        best_fit = None

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        fit = ARIMA(y, order=(p, d, q)).fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                            best_fit = fit
                    except Exception:
                        continue

        if best_fit is None:
            raise ValueError(
                f"auto_arima could not find any converging model within p<={max_p}, "
                f"d<={max_d}, q<={max_q}. Try a wider search range or check the series for issues."
            )

        coefficients = [
            ARIMACoefficient(
                term=name, value=float(best_fit.params[i]),
                std_error=float(best_fit.bse[i]) if best_fit.bse is not None else None,
                p_value=float(best_fit.pvalues[i]) if best_fit.pvalues is not None else None,
            )
            for i, name in enumerate(best_fit.param_names)
        ]

        forecast_points = []
        if forecast_steps > 0:
            fc = best_fit.get_forecast(steps=forecast_steps)
            ci_arr = np.asarray(fc.conf_int(alpha=0.05))
            forecast_points = self._forecast_points(np.asarray(fc.predicted_mean), ci_arr[:, 0], ci_arr[:, 1])

        return ARIMAResult(
            order=list(best_order), coefficients=coefficients, aic=float(best_fit.aic),
            bic=float(best_fit.bic), log_likelihood=float(best_fit.llf),
            fitted_values=np.asarray(best_fit.fittedvalues).tolist(),
            residuals=np.asarray(best_fit.resid).tolist(), original_values=y.tolist(),
            forecast=forecast_points, n=len(y), y_name=column,
        )

    def exponential_smoothing(self, column: str, trend: Optional[str] = None,
                               seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None,
                               forecast_steps: int = 0) -> ExponentialSmoothingResult:
        y = self._get_series(column)
        if trend not in (None, "add", "mul"):
            raise ValueError(f"trend must be None, 'add', or 'mul'. Got '{trend}'.")
        if seasonal not in (None, "add", "mul"):
            raise ValueError(f"seasonal must be None, 'add', or 'mul'. Got '{seasonal}'.")
        if seasonal is not None and not seasonal_periods:
            raise ValueError("seasonal_periods is required when seasonal is specified.")

        try:
            model = ExponentialSmoothing(
                y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit()
        except Exception as e:
            raise ValueError(f"Exponential smoothing failed to fit: {e}") from e

        forecast_points = []
        if forecast_steps > 0:
            fc_mean = np.asarray(fit.forecast(forecast_steps))
            resid_std = float(np.std(fit.resid)) if len(fit.resid) > 1 else 0.0
            z = 1.96
            forecast_points = self._forecast_points(
                fc_mean, fc_mean - z * resid_std, fc_mean + z * resid_std,
            )

        params = fit.params

        def _clean(key):
            val = params.get(key)
            if val is None:
                return None
            try:
                if isinstance(val, float) and np.isnan(val):
                    return None
            except TypeError:
                return None
            return float(val)

        return ExponentialSmoothingResult(
            trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
            smoothing_level=_clean("smoothing_level"), smoothing_trend=_clean("smoothing_trend"),
            smoothing_seasonal=_clean("smoothing_seasonal"), aic=_clean("aic") or (float(fit.aic) if fit.aic is not None and not np.isnan(fit.aic) else None),
            fitted_values=np.asarray(fit.fittedvalues).tolist(), original_values=y.tolist(),
            forecast=forecast_points, n=len(y), y_name=column,
        )

    def garch(self, column: str, p: int = 1, q: int = 1, forecast_steps: int = 0) -> GARCHResult:
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("GARCH modeling requires the 'arch' package. Install with: pip install arch")

        y = self._get_series(column)
        if p < 1 or q < 1:
            raise ValueError("GARCH p and q must both be at least 1.")

        try:
            model = arch_model(y, vol="Garch", p=p, q=q, mean="Zero")
            fit = model.fit(disp="off")
        except Exception as e:
            raise ValueError(f"GARCH({p},{q}) failed to fit: {e}") from e

        params = fit.params
        omega = float(params.get("omega", 0.0))
        alpha_vals = [float(params[k]) for k in params.index if k.startswith("alpha")]
        beta_vals = [float(params[k]) for k in params.index if k.startswith("beta")]
        persistence = sum(alpha_vals) + sum(beta_vals)

        forecast_variance = []
        if forecast_steps > 0:
            fc = fit.forecast(horizon=forecast_steps, reindex=False)
            forecast_variance = np.asarray(fc.variance.values[-1]).tolist()

        return GARCHResult(
            omega=omega, alpha=alpha_vals, beta=beta_vals, persistence=float(persistence),
            aic=float(fit.aic), bic=float(fit.bic),
            conditional_volatility=np.asarray(fit.conditional_volatility).tolist(),
            original_values=y.tolist(), forecast_variance=forecast_variance,
            n=len(y), y_name=column,
        )

    def acf_pacf(self, column: str, n_lags: int = 20) -> ACFResult:
        y = self._get_series(column)
        if n_lags >= len(y):
            raise ValueError(f"n_lags ({n_lags}) must be less than the series length ({len(y)}).")

        acf_vals, acf_ci = acf(y, nlags=n_lags, alpha=0.05)
        pacf_vals, pacf_ci = pacf(y, nlags=n_lags, alpha=0.05)

        return ACFResult(
            acf_values=acf_vals.tolist(), pacf_values=pacf_vals.tolist(),
            acf_confint=np.asarray(acf_ci).tolist(), pacf_confint=np.asarray(pacf_ci).tolist(),
            n_lags=n_lags, n=len(y), column=column,
        )

    def stationarity_test(self, column: str) -> StationarityResult:
        y = self._get_series(column)
        try:
            stat, p_value, _, _, crit_vals, _ = adfuller(y)
        except Exception as e:
            raise ValueError(f"Augmented Dickey-Fuller test failed: {e}") from e

        return StationarityResult(
            statistic=float(stat), p_value=float(p_value), is_stationary=bool(p_value < 0.05),
            critical_values={k: float(v) for k, v in crit_vals.items()}, n=len(y), column=column,
        )
