from typing import List, Optional
import numpy as np
from spssmirror.core._engine import DataEngine
from spssmirror.core._formula import parse_formula
from spssmirror.preprocessing._validators import validate_numeric_column, validate_grouping_column, validate_min_sample_size
from spssmirror.models._results import (
    PosteriorParameter, BayesianTestResult, BayesianRegressionResult, DataQuality,
)

_SAMPLE_KWARGS = dict(cores=1, blas_cores=1, progressbar=False)


def _require_pymc():
    """Lazy-import pymc with a clear error message. Kept out of the
    module-level imports deliberately: pymc is a heavy, optional
    dependency (pip install spssmirror[bayesian]), and only
    bayesian_ttest()/bayesian_linear_regression() actually need it --
    bayesian_proportion_test() is closed-form and works without it. A
    top-level `import pymc` here would make the ENTIRE spssmirror package
    fail to import for anyone who hasn't installed the [bayesian] extra,
    since statistics/__init__.py imports every engine unconditionally."""
    try:
        import pymc as pm
        return pm
    except ImportError:
        raise ImportError(
            "This method requires the 'pymc' package. "
            "Install with: pip install spssmirror[bayesian]"
        )


def _hdi(samples: np.ndarray, prob: float = 0.95) -> tuple:
    """Highest-density interval, computed directly rather than via arviz
    (whose az.hdi() kwarg name has changed across versions — this is a
    stable, dependency-free implementation instead)."""
    s = np.sort(samples)
    n = len(s)
    interval_idx_inc = int(np.floor(prob * n))
    n_intervals = max(n - interval_idx_inc, 1)
    interval_width = s[interval_idx_inc:] - s[:n_intervals]
    min_idx = int(np.argmin(interval_width))
    return float(s[min_idx]), float(s[min_idx + interval_idx_inc])


def _rhat(chain_samples: np.ndarray) -> Optional[float]:
    """Gelman-Rubin R-hat from a (chains, draws) array. Returns None if
    only one chain is available (R-hat is undefined for a single chain)."""
    if chain_samples.ndim != 2 or chain_samples.shape[0] < 2:
        return None
    m, n = chain_samples.shape
    chain_means = chain_samples.mean(axis=1)
    chain_vars = chain_samples.var(axis=1, ddof=1)
    grand_mean = chain_means.mean()
    b = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    w = chain_vars.mean()
    if w == 0:
        return None
    var_hat = (1 - 1 / n) * w + b / n
    return float(np.sqrt(var_hat / w))


def _summarize(idata, name: str, hdi_prob: float) -> PosteriorParameter:
    post = idata.posterior[name].values  # shape (chains, draws, ...)
    flat = post.ravel()
    r_hat = _rhat(post) if post.ndim == 2 else None
    lo, hi = _hdi(flat, hdi_prob)

    return PosteriorParameter(
        name=name, mean=float(flat.mean()), std=float(flat.std(ddof=1)),
        hdi_lower=lo, hdi_upper=hi, r_hat=r_hat, ess_bulk=None, samples=flat.tolist(),
    )


class BayesianEngine:
    """
    Bayesian counterparts to the frequentist t-test/regression suite,
    built on PyMC's NUTS sampler. PyMC/arviz internals never leave this
    module — every method returns a SPSSMirror result model.

    Runtime note: every method fits an MCMC model, so calls here take a
    few seconds (typically 3-15s for the default 1000 tuning + 1000
    sampling draws across 2 chains), unlike the closed-form frequentist
    engines. `cores=1` is pinned deliberately — auto core-detection
    (`blas_cores='auto'`) can raise a ZeroDivisionError in single-CPU
    sandboxed environments, so both `cores` and `blas_cores` are fixed
    explicitly rather than left to autodetection.
    """

    def __init__(self, engine: DataEngine):
        self._engine = engine

    def _quality(self, n_original: int, n_used: int) -> DataQuality:
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def _sample(self, model, draws: int, tune: int, chains: int, seed: int):
        pm = _require_pymc()
        with model:
            idata = pm.sample(draws, tune=tune, chains=chains, random_seed=seed, **_SAMPLE_KWARGS)
        return idata

    def _converged(self, params: List[PosteriorParameter]) -> bool:
        rhats = [p.r_hat for p in params if p.r_hat is not None]
        return all(r < 1.05 for r in rhats) if rhats else True

    def bayesian_ttest(self, value_col: str, group_col: str, group1, group2,
                        draws: int = 1000, tune: int = 1000, chains: int = 2,
                        hdi_prob: float = 0.95, seed: int = 0) -> BayesianTestResult:
        value_series = self._engine.get_column(value_col)
        group_series = self._engine.get_column(group_col)
        validate_numeric_column(value_series, value_col)
        validate_grouping_column(group_series, group_col, min_groups=2)
        n_original = self._engine.shape()[0]

        mask = ~(value_series.isna() | group_series.isna())
        v = value_series[mask].to_numpy(dtype=float)
        g = group_series[mask].to_numpy()
        v1, v2 = v[g == group1], v[g == group2]
        validate_min_sample_size(len(v1), 3, f"Bayesian t-test for group '{group1}'")
        validate_min_sample_size(len(v2), 3, f"Bayesian t-test for group '{group2}'")

        pooled_sd = float(np.std(np.concatenate([v1, v2]), ddof=1))
        overall_mean = float(np.mean(np.concatenate([v1, v2])))

        pm = _require_pymc()
        try:
            with pm.Model() as model:
                mu1 = pm.Normal("mu1", mu=overall_mean, sigma=pooled_sd * 5)
                mu2 = pm.Normal("mu2", mu=overall_mean, sigma=pooled_sd * 5)
                sigma1 = pm.HalfNormal("sigma1", sigma=pooled_sd * 3)
                sigma2 = pm.HalfNormal("sigma2", sigma=pooled_sd * 3)
                pm.Normal("obs1", mu=mu1, sigma=sigma1, observed=v1)
                pm.Normal("obs2", mu=mu2, sigma=sigma2, observed=v2)
                pm.Deterministic("diff", mu1 - mu2)
                idata = self._sample(model, draws, tune, chains, seed)
        except Exception as e:
            raise ValueError(f"Bayesian t-test failed to sample: {e}") from e

        params = [_summarize(idata, name, hdi_prob) for name in ("mu1", "mu2", "diff")]
        diff_samples = np.asarray(params[2].samples)
        prob_gt_0 = float((diff_samples > 0).mean())
        direction_label = f"P({group1} > {group2})" if prob_gt_0 >= 0.5 else f"P({group1} < {group2})"
        direction_value = prob_gt_0 if prob_gt_0 >= 0.5 else 1 - prob_gt_0

        bf10 = self._savage_dickey_bf(diff_samples, prior_sd=pooled_sd * 5 * np.sqrt(2))

        return BayesianTestResult(
            test_name="Bayesian Independent T-Test", parameters=params,
            prob_direction=direction_value, prob_direction_label=direction_label,
            bayes_factor_10=bf10, hdi_prob=hdi_prob, n=len(v1) + len(v2),
            n_draws=draws, n_chains=chains, converged=self._converged(params),
            data_quality=self._quality(n_original, len(v1) + len(v2)),
        )

    def bayesian_proportion_test(self, successes: int, trials: int, prior_alpha: float = 1.0,
                                  prior_beta: float = 1.0, hdi_prob: float = 0.95) -> BayesianTestResult:
        """Closed-form Beta-Binomial conjugate update — no MCMC needed, so
        this returns instantly rather than taking several seconds."""
        if trials <= 0:
            raise ValueError("trials must be positive.")
        if not (0 <= successes <= trials):
            raise ValueError(f"successes must be between 0 and trials. Got {successes}/{trials}.")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must both be positive.")

        post_alpha = prior_alpha + successes
        post_beta = prior_beta + (trials - successes)

        rng = np.random.default_rng(0)
        samples = rng.beta(post_alpha, post_beta, size=20000)
        lo, hi = _hdi(samples, hdi_prob)
        param = PosteriorParameter(
            name="p", mean=float(samples.mean()), std=float(samples.std(ddof=1)),
            hdi_lower=lo, hdi_upper=hi, r_hat=None, ess_bulk=None, samples=samples.tolist(),
        )

        return BayesianTestResult(
            test_name="Bayesian Proportion Test (Beta-Binomial)", parameters=[param],
            prob_direction=float((samples > 0.5).mean()), prob_direction_label="P(p > 0.5)",
            bayes_factor_10=None, hdi_prob=hdi_prob, n=trials, n_draws=len(samples),
            n_chains=1, converged=True,
            data_quality=DataQuality(n_rows_original=trials, n_rows_analyzed=trials,
                                      n_nulls_dropped=0, max_missing_ratio=0.0),
        )

    def bayesian_linear_regression(self, formula: str, draws: int = 1000, tune: int = 1000,
                                    chains: int = 2, hdi_prob: float = 0.95, seed: int = 0) -> BayesianRegressionResult:
        parsed = parse_formula(formula, self._engine.to_dataframe())
        n_original = self._engine.shape()[0]
        y_sd = float(np.std(parsed.y, ddof=1)) if parsed.n_obs > 1 else 1.0

        pm = _require_pymc()
        try:
            with pm.Model() as model:
                betas = pm.Normal("betas", mu=0, sigma=y_sd * 10, shape=parsed.X.shape[1])
                sigma = pm.HalfNormal("sigma", sigma=y_sd * 3)
                mu = pm.math.dot(parsed.X, betas)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=parsed.y)
                idata = self._sample(model, draws, tune, chains, seed)
        except Exception as e:
            raise ValueError(f"Bayesian regression failed to sample: {e}") from e

        params = []
        betas_post = idata.posterior["betas"].values  # (chains, draws, k)
        for i, name in enumerate(parsed.x_names):
            sub = betas_post[:, :, i]
            flat = sub.ravel()
            r_hat = _rhat(sub)
            lo, hi = _hdi(flat, hdi_prob)
            params.append(PosteriorParameter(
                name=name, mean=float(flat.mean()), std=float(flat.std(ddof=1)),
                hdi_lower=lo, hdi_upper=hi, r_hat=r_hat, ess_bulk=None, samples=flat.tolist(),
            ))
        params.append(_summarize(idata, "sigma", hdi_prob))

        return BayesianRegressionResult(
            parameters=params, hdi_prob=hdi_prob, n=parsed.n_obs, n_draws=draws,
            n_chains=chains, converged=self._converged(params), y_name=parsed.y_name,
            data_quality=self._quality(n_original, parsed.n_obs),
        )

    @staticmethod
    def _savage_dickey_bf(diff_samples: np.ndarray, prior_sd: float) -> Optional[float]:
        """Approximate Bayes factor (H1: real difference vs H0: no
        difference) via the Savage-Dickey density ratio: posterior density
        at 0 vs prior density at 0, for the null point diff=0."""
        try:
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(diff_samples)
            posterior_density_at_0 = float(kde(0.0)[0])
            prior_density_at_0 = float(scipy_stats.norm.pdf(0.0, loc=0, scale=prior_sd))
            if posterior_density_at_0 <= 0:
                return None
            return prior_density_at_0 / posterior_density_at_0
        except Exception:
            return None
