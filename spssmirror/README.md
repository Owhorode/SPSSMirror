# SPSSMirror

[![PyPI version](https://img.shields.io/pypi/v/spssmirror.svg)](https://pypi.org/project/spssmirror/)
[![Python versions](https://img.shields.io/pypi/pyversions/spssmirror.svg)](https://pypi.org/project/spssmirror/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SPSSMirror** is a unified, self-contained statistical analysis library for
Python — a modern SPSS/R replacement. Every method wraps
scipy/statsmodels/scikit-learn/lifelines/pymc/arch internally and returns a
typed, immutable result object that **prints as a clean pandas table**. Any
method SPSSMirror doesn't define falls straight through to the underlying
pandas DataFrame, unmodified — so it behaves exactly like pandas until you
need statistics, and then it's one flat method call away.

```python
from spssmirror import SPSSMirror

mirror = SPSSMirror().load_csv("survey.csv")

mirror.head()                                    # plain pandas, unchanged
result = mirror.linear_regression("score ~ age + C(group)")
print(result)                                     # pandas-style table
mirror.hint("regression")                         # see what else is available
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
"result" looks like. SPSSMirror collapses that into one consistent, flat API:

- **One flat method per analysis.** `mirror.t_test_independent(...)` returns
  a result with `.statistic`, `.p_value`, `.effect_size`, and
  `.data_quality` — every method across every domain follows the same shape,
  and none of them are nested behind a sub-object.
- **Prints like pandas.** Every result renders as a real pandas `Series` (or
  `DataFrame`, for multi-row results like regression coefficients) — no
  manual formatting, `print(result)` just works.
- **Full pandas underneath.** `mirror.groupby(...)`, `.fillna(...)`,
  `.dtypes`, `.merge(...)` — anything not defined by SPSSMirror passes
  straight through to the real pandas DataFrame, untouched.
- **Discoverable.** `mirror.hint()` lists every statistical method available,
  as a DataFrame, with a one-line description each.
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

Four domains depend on heavier, optional libraries and are installed as
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

Call `mirror.hint()` at any time to see this same table live, with every
method available on your installed version. Methods are grouped below just
for reference — on `mirror` itself, every one is a flat, single call
(`mirror.pearson(...)`, not `mirror.correlations().pearson(...)`).

| Domain | Representative methods |
|---|---|
| Descriptive | `summary`, `frequency_table`, `crosstab` |
| Regression | `linear_regression`, `logistic_regression`, `poisson_regression`, `glm`, `robust_regression`, `ridge_regression`, `lasso_regression`, `elastic_net_regression` |
| Frequentist (parametric) | `t_test_one_sample`, `t_test_independent`, `t_test_paired`, `anova_oneway`, `anova_twoway`, `ancova`, `anova_repeated_measures`, `manova` |
| Frequentist (non-parametric) | `mann_whitney_u`, `wilcoxon_signed_rank`, `kruskal_wallis`, `friedman_test` |
| Categorical | `chi_square_independence`, `fishers_exact`, `mcnemar_test` |
| Correlation | `pearson`, `spearman`, `kendall_tau`, `point_biserial`, `partial`, `correlation_matrix` |
| Psychometrics | `cronbach_alpha`, `mcdonald_omega`, `split_half`, `kmo`, `bartlett_sphericity`, `item_analysis`, `efa` |
| Effect sizes | `cohens_d`, `hedges_g`, `glass_delta`, `eta_squared`, `omega_squared`, `cramers_v`, `odds_ratio` |
| Power analysis | `power_ttest_independent`, `power_anova`, `power_correlation`, `power_chisquare`, `power_curve_ttest`, `power_curve_anova`, `power_curve_correlation` |
| Diagnostics | `normality_tests`, `homogeneity_of_variance`, `vif`, `residual_diagnostics`, `outliers` |
| Mixed models | `linear_mixed_model` (random intercept/slope, ICC) |
| Bayesian *(extra)* | `bayesian_ttest`, `bayesian_proportion_test`, `bayesian_linear_regression` |
| Time series *(extra)* | `arima`, `auto_arima`, `exponential_smoothing`, `garch`, `acf_pacf`, `stationarity_test` |
| Survival *(extra)* | `kaplan_meier`, `logrank_test`, `cox_ph`, `parametric_survival` |
| Multivariate *(extra)* | `pca`, `kmeans_clustering`, `hierarchical_clustering`, `linear_discriminant`, `quadratic_discriminant`, `canonical_correlation` |

The old grouped accessors (`.regression()`, `.frequentist()`, etc.) still
work identically for anyone with existing code — nothing was removed, the
flat methods are additive.

Every result is a frozen [Pydantic](https://docs.pydantic.dev) model under
the hood — call `.to_frame()` to get the pandas Series/DataFrame explicitly,
or `.model_dump()` / `.model_dump_json()` to export raw values.

## Quick start

```python
from spssmirror import SPSSMirror

mirror = SPSSMirror().load_csv("data.csv")

mirror.head()                    # normal pandas -- works immediately
mirror.dtypes
mirror.fillna(0)

# Reliability
print(mirror.cronbach_alpha(["q1", "q2", "q3", "q4"]))

# Group comparison with effect size
print(mirror.t_test_independent("score", "group", "A", "B"))

# Regression -- no statsmodels import needed anywhere in your code
reg = mirror.linear_regression("outcome ~ predictor1 + C(category)")
print(reg.to_frame())            # coefficients as a pandas DataFrame

# Power analysis
power = mirror.power_ttest_independent(effect_size=0.5, alpha=0.05, power=0.80)
print(f"Need {power.n:.0f} participants per group")

# Not sure what's available?
mirror.hint()                    # everything
mirror.hint("regression")        # just one group
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
   SPSSMirror's own typed models (which print like pandas).
2. **Honest statistics over convenient statistics.** If a number can't be
   computed validly, the field is `None`, not a fabricated or silently
   wrong value.
3. **Data quality is never hidden.** Every result that drops rows (nulls,
   non-finite values) reports exactly how many and what fraction.
4. **Pandas stays pandas.** SPSSMirror adds statistics on top of a
   DataFrame; it never hides or replaces pandas' own methods.
5. **No visualization dependency.** Results are plain, inspectable data —
   pair with whatever plotting library your project already uses.

## Testing

```bash
git clone https://github.com/<your-username>/spssmirror.git
cd spssmirror
pip install -e ".[all,dev]"
pytest tests/ -v
```

The test suite checks every method against **engineered ground truth**
(known true effects and known coefficients, not just "does it run") — see
`tests/conftest.py` for the fixtures.

## Contributing

Issues and pull requests are welcome. Please include a test demonstrating
the bug or feature — see `tests/` for the existing pattern (each test
targets one method against either a known analytical result or a clearly
engineered scenario).

## License

MIT — see [LICENSE](LICENSE).