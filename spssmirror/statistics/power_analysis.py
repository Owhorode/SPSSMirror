from typing import Optional
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower, GofChisquarePower
from spssmirror.models._results import PowerResult, PowerCurveResult


def _validate_alpha(alpha: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1. Got {alpha}.")


def _exactly_one_none(*args) -> None:
    n_none = sum(1 for a in args if a is None)
    if n_none != 1:
        raise ValueError(
            f"Exactly one of the solvable parameters must be left as None (to be solved for). "
            f"Got {len(args) - n_none} provided and {n_none} missing."
        )


class PowerAnalysisEngine:
    """
    Power / sample-size analysis for every test family built so far.
    Leave exactly one of the relevant parameters as None and it is solved
    for — e.g. give effect_size + n + alpha to get power, or effect_size +
    power + alpha to get required n.
    """

    def power_ttest_independent(self, effect_size: float, n: Optional[float] = None,
                                 alpha: float = 0.05, power: Optional[float] = None) -> PowerResult:
        _validate_alpha(alpha)
        _exactly_one_none(n, power)
        analysis = TTestIndPower()
        if power is None:
            power = float(analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0))
        else:
            n = float(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0))

        return PowerResult(
            test_type="Independent T-Test", effect_size=effect_size, alpha=alpha,
            power=power, n=n, details={"n_per_group": n, "total_n": n * 2 if n else None},
        )

    def power_ttest_paired(self, effect_size: float, n: Optional[float] = None,
                            alpha: float = 0.05, power: Optional[float] = None) -> PowerResult:
        _validate_alpha(alpha)
        _exactly_one_none(n, power)
        analysis = TTestPower()
        if power is None:
            power = float(analysis.power(effect_size=effect_size, nobs=n, alpha=alpha))
        else:
            n = float(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power))

        return PowerResult(test_type="Paired T-Test", effect_size=effect_size, alpha=alpha, power=power, n=n)

    def power_ttest_one_sample(self, effect_size: float, n: Optional[float] = None,
                                alpha: float = 0.05, power: Optional[float] = None) -> PowerResult:
        _validate_alpha(alpha)
        _exactly_one_none(n, power)
        analysis = TTestPower()
        if power is None:
            power = float(analysis.power(effect_size=effect_size, nobs=n, alpha=alpha))
        else:
            n = float(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power))

        return PowerResult(test_type="One-Sample T-Test", effect_size=effect_size, alpha=alpha, power=power, n=n)

    def power_anova(self, effect_size: float, k_groups: int, n_per_group: Optional[float] = None,
                     alpha: float = 0.05, power: Optional[float] = None) -> PowerResult:
        """Note: statsmodels' underlying FTestAnovaPower works in terms of
        TOTAL sample size across all groups; this method converts to/from
        per-group n so the public API matches how researchers plan designs."""
        _validate_alpha(alpha)
        if k_groups < 2:
            raise ValueError("k_groups must be at least 2.")
        _exactly_one_none(n_per_group, power)
        analysis = FTestAnovaPower()
        if power is None:
            total_n = n_per_group * k_groups
            power = float(analysis.power(effect_size=effect_size, nobs=total_n, alpha=alpha, k_groups=k_groups))
        else:
            total_n = float(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups))
            n_per_group = total_n / k_groups

        return PowerResult(
            test_type="One-Way ANOVA", effect_size=effect_size, alpha=alpha, power=power, n=n_per_group,
            details={"k_groups": k_groups, "n_per_group": n_per_group, "total_n": total_n},
        )

    def power_correlation(self, effect_size: float, n: Optional[float] = None,
                           alpha: float = 0.05, power: Optional[float] = None) -> PowerResult:
        """Uses the standard Fisher z-transform normal approximation (as in
        R's pwr.r.test) rather than an exact noncentral distribution."""
        _validate_alpha(alpha)
        if not (-1 < effect_size < 1):
            raise ValueError("effect_size (r) must be strictly between -1 and 1.")
        _exactly_one_none(n, power)

        z_r = np.arctanh(effect_size)
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)

        if power is None:
            if n <= 3:
                raise ValueError("n must be greater than 3 for correlation power analysis.")
            power = float(scipy_stats.norm.cdf(abs(z_r) * np.sqrt(n - 3) - z_alpha))
        else:
            z_beta = scipy_stats.norm.ppf(power)
            n = float(((z_alpha + z_beta) / abs(z_r)) ** 2 + 3)

        return PowerResult(test_type="Correlation (Pearson)", effect_size=effect_size, alpha=alpha, power=power, n=n)

    def power_chisquare(self, effect_size: float, df: int, n: Optional[float] = None,
                         alpha: float = 0.05, power: Optional[float] = None) -> PowerResult:
        _validate_alpha(alpha)
        if df < 1:
            raise ValueError("df must be at least 1.")
        _exactly_one_none(n, power)
        analysis = GofChisquarePower()
        n_bins = df + 1
        if power is None:
            power = float(analysis.power(effect_size=effect_size, nobs=n, alpha=alpha, n_bins=n_bins))
        else:
            n = float(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, n_bins=n_bins))

        return PowerResult(
            test_type="Chi-Square", effect_size=effect_size, alpha=alpha, power=power, n=n, details={"df": df},
        )

    # ------------------- power curves -------------------

    def power_curve_ttest(self, effect_size: float, alpha: float = 0.05,
                           n_min: int = 5, n_max: int = 200, n_points: int = 60,
                           target_power: float = 0.8) -> PowerCurveResult:
        _validate_alpha(alpha)
        analysis = TTestIndPower()
        n_values = np.linspace(n_min, n_max, n_points)
        powers = [float(analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0)) for n in n_values]

        return PowerCurveResult(
            test_type="Independent T-Test", x_label="Sample size per group",
            x_values=n_values.tolist(), y_values=powers, target_power=target_power,
            context={"effect_size": effect_size, "alpha": alpha},
        )

    def power_curve_anova(self, effect_size: float, k_groups: int, alpha: float = 0.05,
                           n_min: int = 5, n_max: int = 200, n_points: int = 60,
                           target_power: float = 0.8) -> PowerCurveResult:
        """x-axis is per-group n; statsmodels internally wants total nobs."""
        _validate_alpha(alpha)
        analysis = FTestAnovaPower()
        n_values = np.linspace(n_min, n_max, n_points)
        powers = [float(analysis.power(effect_size=effect_size, nobs=n * k_groups, alpha=alpha, k_groups=k_groups))
                  for n in n_values]

        return PowerCurveResult(
            test_type="One-Way ANOVA", x_label="Sample size per group",
            x_values=n_values.tolist(), y_values=powers, target_power=target_power,
            context={"effect_size": effect_size, "alpha": alpha, "k_groups": k_groups},
        )

    def power_curve_correlation(self, effect_size: float, alpha: float = 0.05,
                                 n_min: int = 5, n_max: int = 200, n_points: int = 60,
                                 target_power: float = 0.8) -> PowerCurveResult:
        _validate_alpha(alpha)
        z_r = np.arctanh(effect_size)
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        n_values = np.linspace(max(n_min, 4), n_max, n_points)
        powers = [float(scipy_stats.norm.cdf(abs(z_r) * np.sqrt(n - 3) - z_alpha)) for n in n_values]

        return PowerCurveResult(
            test_type="Correlation (Pearson)", x_label="Sample size",
            x_values=n_values.tolist(), y_values=powers, target_power=target_power,
            context={"effect_size": effect_size, "alpha": alpha},
        )
