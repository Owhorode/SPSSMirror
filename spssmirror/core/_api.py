import inspect
from typing import Dict, List, Any, Optional
import pandas as pd
from spssmirror.core._engine import DataEngine
from spssmirror.core._formula import parse_formula, ParsedFormula
from spssmirror.statistics.descriptive import DescriptiveEngine
from spssmirror.statistics.regression import RegressionEngine
from spssmirror.statistics.frequentist_parametric import FrequentistParametricEngine
from spssmirror.statistics.frequentist_nonparametric import FrequentistNonparametricEngine
from spssmirror.statistics.categorical import CategoricalEngine
from spssmirror.statistics.correlations import CorrelationEngine
from spssmirror.statistics.psychometrics import PsychometricsEngine
from spssmirror.statistics.effect_sizes import EffectSizeEngine
from spssmirror.statistics.power_analysis import PowerAnalysisEngine
from spssmirror.statistics.diagnostics import DiagnosticsEngine
from spssmirror.statistics.mixed_models import MixedModelsEngine
from spssmirror.statistics.bayesian import BayesianEngine
from spssmirror.statistics.timeseries import TimeSeriesEngine
from spssmirror.statistics.survival import SurvivalEngine
from spssmirror.statistics.multivariate import MultivariateEngine

# Renames applied when a grouped-engine method is flattened onto SPSSMirror
# directly -- mostly to disambiguate regression() variants now that they no
# longer sit inside a `.regression()` namespace giving them context.
_RENAME = {
    ("regression", "linear"): "linear_regression",
    ("regression", "logistic"): "logistic_regression",
    ("regression", "poisson"): "poisson_regression",
    ("regression", "robust"): "robust_regression",
    ("regression", "ridge"): "ridge_regression",
    ("regression", "lasso"): "lasso_regression",
    ("regression", "elastic_net"): "elastic_net_regression",
    ("regression", "linear_columns"): "linear_regression_columns",
}

# Short one-line descriptions for .hint(). Anything not listed here still
# works fine -- hint() just shows a generic fallback description for it.
_DESCRIPTIONS = {
    "summary": "Descriptive statistics for one column (mean, sd, min, max...)",
    "frequency_table": "Frequency/percentage table for one column",
    "crosstab": "Cross-tabulation of two columns, with chi-square",
    "linear_regression": "Ordinary least squares linear regression",
    "logistic_regression": "Logistic regression for a binary outcome",
    "poisson_regression": "Poisson regression for count outcomes",
    "glm": "Generalized linear model (choose the family)",
    "robust_regression": "Linear regression resistant to outliers",
    "ridge_regression": "Ridge (L2-regularized) regression",
    "lasso_regression": "Lasso (L1-regularized) regression",
    "elastic_net_regression": "Elastic net (L1+L2) regression",
    "t_test_one_sample": "Compare one column's mean to a fixed value",
    "t_test_independent": "Compare means of two independent groups",
    "t_test_paired": "Compare means of two paired/matched columns",
    "anova_oneway": "Compare means across 3+ independent groups",
    "anova_twoway": "Two-factor ANOVA with interaction",
    "ancova": "ANOVA with a continuous covariate",
    "anova_repeated_measures": "ANOVA for repeated within-subject measures",
    "manova": "Multivariate ANOVA (several dependent variables)",
    "mann_whitney_u": "Non-parametric alternative to the independent t-test",
    "wilcoxon_signed_rank": "Non-parametric alternative to the paired t-test",
    "kruskal_wallis": "Non-parametric alternative to one-way ANOVA",
    "friedman_test": "Non-parametric alternative to repeated-measures ANOVA",
    "chi_square_independence": "Test association between two categorical columns",
    "fishers_exact": "Exact test for a 2x2 categorical table",
    "mcnemar_test": "Test for paired categorical (before/after) data",
    "pearson": "Pearson correlation between two numeric columns",
    "spearman": "Spearman rank correlation between two columns",
    "kendall_tau": "Kendall's Tau rank correlation",
    "point_biserial": "Correlation between a binary and a continuous column",
    "partial": "Correlation between two columns, controlling for others",
    "correlation_matrix": "Correlation matrix across several columns",
    "cronbach_alpha": "Reliability (internal consistency) of a set of items",
    "mcdonald_omega": "McDonald's Omega reliability estimate",
    "split_half": "Split-half reliability estimate",
    "kmo": "Kaiser-Meyer-Olkin sampling adequacy for factor analysis",
    "bartlett_sphericity": "Bartlett's test of sphericity for factor analysis",
    "item_analysis": "Per-item statistics for a scale (item-total correlation etc.)",
    "efa": "Exploratory factor analysis",
    "cohens_d": "Cohen's d effect size between two groups",
    "hedges_g": "Hedges' g (bias-corrected) effect size",
    "glass_delta": "Glass's Delta effect size",
    "eta_squared": "Eta-squared effect size for group differences",
    "omega_squared": "Omega-squared (less biased) effect size",
    "cramers_v": "Cramer's V effect size for categorical association",
    "odds_ratio": "Odds ratio for a 2x2 categorical table",
    "power_ttest_independent": "Power/sample size for an independent t-test",
    "power_ttest_paired": "Power/sample size for a paired t-test",
    "power_ttest_one_sample": "Power/sample size for a one-sample t-test",
    "power_anova": "Power/sample size for one-way ANOVA",
    "power_correlation": "Power/sample size for a correlation test",
    "power_chisquare": "Power/sample size for a chi-square test",
    "power_curve_ttest": "Power across a range of sample sizes (t-test)",
    "power_curve_anova": "Power across a range of sample sizes (ANOVA)",
    "power_curve_correlation": "Power across a range of sample sizes (correlation)",
    "normality_tests": "Shapiro-Wilk/KS/etc. tests of normality for one column",
    "homogeneity_of_variance": "Levene's/Bartlett's test of equal variances",
    "vif": "Variance Inflation Factor -- multicollinearity check",
    "residual_diagnostics": "Leverage/Cook's distance diagnostics for a regression",
    "outliers": "Detect outliers in one column (IQR or z-score)",
    "linear_mixed_model": "Mixed-effects model with random intercept/slope",
    "bayesian_ttest": "Bayesian version of the independent t-test",
    "bayesian_proportion_test": "Bayesian test for a single proportion",
    "bayesian_linear_regression": "Bayesian linear regression",
    "arima": "Fit an ARIMA time-series model",
    "auto_arima": "Automatically select the best ARIMA order",
    "exponential_smoothing": "Exponential smoothing forecast model",
    "garch": "GARCH volatility model",
    "acf_pacf": "Autocorrelation / partial autocorrelation of a series",
    "stationarity_test": "Augmented Dickey-Fuller stationarity test",
    "kaplan_meier": "Kaplan-Meier survival curve",
    "logrank_test": "Log-rank test comparing survival curves",
    "cox_ph": "Cox proportional hazards regression",
    "parametric_survival": "Parametric survival model (Weibull etc.)",
    "pca": "Principal component analysis",
    "kmeans_clustering": "K-means clustering",
    "hierarchical_clustering": "Hierarchical (agglomerative) clustering",
    "linear_discriminant": "Linear discriminant analysis (classification)",
    "quadratic_discriminant": "Quadratic discriminant analysis (classification)",
    "canonical_correlation": "Canonical correlation between two variable sets",
}


class SPSSMirror:
    """
    Central facade. Every statistical method lives directly on this class
    (flat, one dot: `mirror.linear_regression(...)`), and any attribute or
    method not defined here or above falls through to the underlying
    pandas DataFrame automatically (`mirror.head()`, `mirror.dtypes`,
    `mirror.groupby(...)`, `mirror.fillna(...)` -- the full, completely
    unmodified pandas API). Call `mirror.hint()` to see every available
    statistical method, or `mirror.hint("regression")` to filter by group.
    """

    def __init__(self):
        self._engine = DataEngine()
        engines = {
            "descriptive": DescriptiveEngine(self._engine),
            "regression": RegressionEngine(self._engine),
            "frequentist": FrequentistParametricEngine(self._engine),
            "nonparametric": FrequentistNonparametricEngine(self._engine),
            "categorical": CategoricalEngine(self._engine),
            "correlations": CorrelationEngine(self._engine),
            "psychometrics": PsychometricsEngine(self._engine),
            "effect_sizes": EffectSizeEngine(self._engine),
            "power": PowerAnalysisEngine(),
            "diagnostics": DiagnosticsEngine(self._engine),
            "mixed_models": MixedModelsEngine(self._engine),
            "bayesian": BayesianEngine(self._engine),
            "timeseries": TimeSeriesEngine(self._engine),
            "survival": SurvivalEngine(self._engine),
            "multivariate": MultivariateEngine(self._engine),
        }
        # Keep the original grouped accessors too (mirror.regression()...) --
        # fully backward compatible, nothing removed.
        for group_name, engine_obj in engines.items():
            setattr(self, f"_{group_name}_engine", engine_obj)

        self._method_registry = []
        known_names = set(dir(type(self))) | set(self.__dict__.keys())
        for group_name, engine_obj in engines.items():
            for name, method in inspect.getmembers(engine_obj, predicate=inspect.ismethod):
                if name.startswith("_"):
                    continue
                flat_name = _RENAME.get((group_name, name), name)
                if flat_name in known_names:
                    continue  # never overwrite a real SPSSMirror method (load_csv, shape, etc.)
                setattr(self, flat_name, method)
                known_names.add(flat_name)
                desc = _DESCRIPTIONS.get(flat_name, f"See help({group_name} engine) for details.")
                self._method_registry.append((flat_name, group_name, desc))

    def __getattr__(self, name):
        # Only triggered when normal attribute lookup (including everything
        # set in __init__ above) finds nothing -- so this never shadows a
        # real SPSSMirror or statistical method, only fills genuine gaps
        # with the underlying pandas DataFrame's own methods/attributes.
        # Must raise AttributeError (not ValueError) on failure, or Python's
        # own attribute protocol (hasattr, getattr with a default, etc.)
        # breaks.
        engine = object.__getattribute__(self, "_engine")
        try:
            df = engine.to_dataframe()
        except ValueError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(and no data is loaded yet to check pandas for it either: {e})"
            ) from None
        try:
            return getattr(df, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(also not a pandas DataFrame method/attribute)."
            ) from None

    def load_dict(self, data: Dict[str, List[Any]]) -> "SPSSMirror":
        self._engine.load_dict(data)
        return self

    def load_dataframe(self, df: pd.DataFrame) -> "SPSSMirror":
        self._engine.load_dataframe(df)
        return self

    def load_csv(self, filepath: str, **kwargs) -> "SPSSMirror":
        self._engine.load_csv(filepath, **kwargs)
        return self

    def load_excel(self, filepath: str, **kwargs) -> "SPSSMirror":
        self._engine.load_excel(filepath, **kwargs)
        return self

    def columns(self) -> tuple:
        return self._engine.columns()

    def dtypes(self) -> Dict[str, str]:
        return self._engine.dtypes()

    def preview(self, n_rows: int = 5) -> str:
        return self._engine.preview(n_rows)

    def shape(self) -> tuple:
        return self._engine.shape()

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a copy of the underlying data. Safe to mutate freely --
        changes here will NOT affect this SPSSMirror instance."""
        return self._engine.to_dataframe().copy()

    def formula(self, formula_str: str) -> ParsedFormula:
        return parse_formula(formula_str, self._engine.to_dataframe())

    def hint(self, group: Optional[str] = None) -> pd.DataFrame:
        """Lists every available statistical method as a pandas DataFrame:
        method name, its group, and a one-line description. Pass a group
        name (e.g. 'regression', 'psychometrics') to filter."""
        rows = self._method_registry if group is None else [
            r for r in self._method_registry if r[1] == group
        ]
        return pd.DataFrame(rows, columns=["method", "group", "description"]).set_index("method")

    def descriptive(self) -> DescriptiveEngine:
        return self._descriptive_engine

    def regression(self) -> RegressionEngine:
        return self._regression_engine

    def frequentist(self) -> FrequentistParametricEngine:
        return self._frequentist_engine

    def nonparametric(self) -> FrequentistNonparametricEngine:
        return self._nonparametric_engine

    def categorical(self) -> CategoricalEngine:
        return self._categorical_engine

    def correlations(self) -> CorrelationEngine:
        return self._correlation_engine

    def psychometrics(self) -> PsychometricsEngine:
        return self._psychometrics_engine

    def effect_sizes(self) -> EffectSizeEngine:
        return self._effect_size_engine

    def power(self) -> PowerAnalysisEngine:
        return self._power_engine

    def diagnostics(self) -> DiagnosticsEngine:
        return self._diagnostics_engine

    def mixed_models(self) -> MixedModelsEngine:
        return self._mixed_models_engine

    def bayesian(self) -> BayesianEngine:
        return self._bayesian_engine

    def timeseries(self) -> TimeSeriesEngine:
        return self._timeseries_engine

    def survival(self) -> SurvivalEngine:
        return self._survival_engine

    def multivariate(self) -> MultivariateEngine:
        return self._multivariate_engine

    @property
    def _data_engine(self) -> DataEngine:
        return self._engine
