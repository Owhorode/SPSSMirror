# SPSSMirror

[![PyPI version](https://img.shields.io/pypi/v/spssmirror.svg)](https://pypi.org/project/spssmirror/)
[![Python versions](https://img.shields.io/pypi/pyversions/spssmirror.svg)](https://pypi.org/project/spssmirror/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SPSSMirror** is a unified, self-contained statistical analysis library for
Python. It merges statistical libraries under one umbrella. Every method wraps
scipy/statsmodels/scikit-learn/lifelines/pymc/arch internally and returns a
typed, immutable result object with the statistic, p-value, effect size, and
confidence interval already computed. **You never need to import those
libraries yourself** to get a complete answer.

```python
from spssmirror import SPSSMirror

mirror = SPSSMirror().load_csv("survey.csv")
result = mirror.regression().linear("score ~ age + C(group)")
print(result.r_squared, result.coefficients)
```

---

## Table of contents

- [Why SPSSMirror](#why-spssmirror)
- [Installation](#installation)
- [What's included](#whats-included)
- [Quick start](#quick-start)
- [Design principles](#design-principles)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Why SPSSMirror

Most Python statistics work means juggling `scipy.stats`, `statsmodels`,
`scikit-learn`, and reading each library's own conventions for what a
"result" looks like. SPSSMirror collapses that into one consistent API:

- **One object per analysis.** `mirror.frequentist().t_test_independent(...)`
  returns a `StatTestResult` with `.statistic`, `.p_value`, `.effect_size`,
  and `.data_quality` — every method across every engine follows the same
  shape.
- **Honest about uncertainty.** Regularized regression (Ridge/Lasso) reports
  `std_error`/`p_value` as `None` instead of fabricating classical inference
  that regularization invalidates. A mixed model fit with REML reports
  `aic`/`bic` as `None` rather than silently leaking statsmodels' `NaN`.
- **Refuses to compute nonsense.** Running an ANOVA or regression on a
  column that turns out to be constant raises a clear error instead of a
  false "p = 0.000016, significant!" result caused by floating-point noise
  in the underlying model fit.
- **Every result tracks its own data quality** — `n_rows_original`,
  `n_nulls_dropped`, `max_missing_ratio` — so you always know what was
  silently dropped before you trust a number.
- **Formula syntax where it belongs.** Regression, the ANOVA family, mixed
  models, and residual diagnostics accept R-like formulas via
  [patsy](https://patsy.readthedocs.io): `"y ~ x1 + C(group) * x2"`.

## Installation

```bash
pip install spssmirror
```

This installs the **core** engine — descriptive statistics, regression, the
full frequentist test suite (parametric and non-parametric), categorical
analysis, correlations, psychometrics, effect sizes, power analysis,
diagnostics, and mixed models — with a deliberately lean dependency list
(`pandas`, `numpy`, `scipy`, `statsmodels`, `pydantic`, `patsy`,
`rapidfuzz`, `factor_analyzer`).

Four engines depend on heavier, optional libraries and are installed as
extras:

```bash
pip install spssmirror[bayesian]      # Bayesian t-test/regression (pymc, arviz)
pip install spssmirror[timeseries]    # ARIMA/GARCH forecasting (arch)
pip install spssmirror[survival]      # Kaplan-Meier / Cox PH (lifelines)
pip install spssmirror[multivariate]  # PCA / clustering / discriminant (scikit-learn)
pip install spssmirror[all]           # everything at once
```

The core install works with **zero** optional dependencies present —
verified by installing the built wheel into a clean virtual environment as
part of the test process.

## What's included

| Engine | Access | Methods |
|---|---|---|
| Descriptive | `.descriptive()` | `summary`, `frequency_table`, `crosstab` |
| Regression | `.regression()` | `linear`, `logistic`, `poisson`, `glm`, `robust`, `ridge`, `lasso`, `elastic_net` |
| Frequentist (parametric) | `.frequentist()` | `t_test_one_sample`, `t_test_independent`, `t_test_paired`, `anova_oneway`, `anova_twoway`, `ancova`, `anova_repeated_measures`, `manova` |
| Frequentist (non-parametric) | `.nonparametric()` | `mann_whitney_u`, `wilcoxon_signed_rank`, `kruskal_wallis`, `friedman_test` |
| Categorical | `.categorical()` | `chi_square_independence`, `fishers_exact`, `mcnemar_test` |
| Correlation | `.correlations()` | `pearson`, `spearman`, `kendall_tau`, `point_biserial`, `partial`, `correlation_matrix` |
| Psychometrics | `.psychometrics()` | `cronbach_alpha`, `mcdonald_omega`, `split_half`, `kmo`, `bartlett_sphericity`, `item_analysis`, `efa` |
| Effect sizes | `.effect_sizes()` | `cohens_d`, `hedges_g`, `glass_delta`, `eta_squared`, `omega_squared`, `cramers_v`, `odds_ratio` |
| Power analysis | `.power()` | `power_ttest_independent`, `power_ttest_paired`, `power_ttest_one_sample`, `power_anova`, `power_correlation`, `power_chisquare`, `power_curve_ttest`, `power_curve_anova`, `power_curve_correlation` |
| Diagnostics | `.diagnostics()` | `normality_tests`, `homogeneity_of_variance`, `vif`, `residual_diagnostics`, `outliers` |
| Mixed models | `.mixed_models()` | `linear_mixed_model` (random intercept/slope, ICC) |
| Bayesian *(extra)* | `.bayesian()` | `bayesian_ttest`, `bayesian_proportion_test`, `bayesian_linear_regression` |
| Time series *(extra)* | `.timeseries()` | `arima`, `auto_arima`, `exponential_smoothing`, `garch`, `acf_pacf`, `stationarity_test` |
| Survival *(extra)* | `.survival()` | `kaplan_meier`, `logrank_test`, `cox_ph`, `parametric_survival` |
| Multivariate *(extra)* | `.multivariate()` | `pca`, `kmeans_clustering`, `hierarchical_clustering`, `linear_discriminant`, `quadratic_discriminant`, `canonical_correlation` |

Every method returns a frozen [Pydantic](https://docs.pydantic.dev) model.
Inspect fields directly, or call `.model_dump()` / `.model_dump_json()` to
export.

## Quick start

```python
from spssmirror import SPSSMirror

mirror = SPSSMirror().load_csv("data.csv")

# Reliability
alpha = mirror.psychometrics().cronbach_alpha(["q1", "q2", "q3", "q4"])
print(alpha.statistic)

# Group comparison with effect size
t = mirror.frequentist().t_test_independent("score", "group", "A", "B")
print(t.statistic, t.p_value, t.effect_size)

# Regression — no statsmodels import needed anywhere in your code
reg = mirror.regression().linear("outcome ~ predictor1 + C(category)")
for coef in reg.coefficients:
    print(coef.term, coef.b, coef.p_value)

# Power analysis
power = mirror.power().power_ttest_independent(effect_size=0.5, alpha=0.05, power=0.80)
print(f"Need {power.n:.0f} participants per group")
```

Loading data:

```python
SPSSMirror().load_csv("data.csv")
SPSSMirror().load_excel("data.xlsx")
SPSSMirror().load_dict({"col1": [...], "col2": [...]})
SPSSMirror().load_dataframe(existing_pandas_df)
```

## Design principles

1. **Nothing leaks.** Public methods never return a raw
   scipy/statsmodels/scikit-learn/pymc/lifelines/arch object — only
   SPSSMirror's own typed models.
2. **Honest statistics over convenient statistics.** If a number can't be
   computed validly, the field is `None`, not a fabricated or silently
   wrong value.
3. **Data quality is never hidden.** Every result that drops rows (nulls,
   non-finite values) reports exactly how many and what fraction.
4. **No visualization dependency.** Results are plain, inspectable data —
   pair with whatever plotting library your project already uses.

## Testing

```bash
git clone https://github.com/<your-username>/spssmirror.git
cd spssmirror
pip install -e ".[all,dev]"
pytest tests/ -v
```

The test suite checks every engine against **engineered ground truth**
(known true effects and known coefficients, not just "does it run") — see
`tests/conftest.py` for the fixtures.

## Contributing

Issues and pull requests are welcome. Please include a test demonstrating
the bug or feature — see `tests/` for the existing pattern (each test
targets one method against either a known analytical result or a clearly
engineered scenario).

## License

MIT.
