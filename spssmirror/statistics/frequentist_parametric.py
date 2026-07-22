from typing import List
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from spssmirror.core._engine import DataEngine
from spssmirror.preprocessing._validators import (
    validate_numeric_column, validate_grouping_column, validate_min_sample_size, validate_has_variance,
)
from spssmirror.models._results import (
    StatTestResult, DataQuality, ANOVAResult, ANOVATerm, PostHocResult,
    GroupSummary, MANOVAResult, MANOVAStatistic,
)


class FrequentistParametricEngine:
    """
    t-test family, ANOVA family (one-way, two-way/factorial, ANCOVA,
    repeated measures), and MANOVA. statsmodels/scipy are used internally
    only — every method returns a SPSSMirror result model.

    Design note: ANOVA tables use Type II sums of squares (statsmodels
    `anova_lm(typ=2)`), which is invariant to factor coding order. SPSS
    defaults to Type III — for balanced designs the two agree; for
    unbalanced designs with interactions they can diverge. This is a
    deliberate, documented choice rather than a silent difference.
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

    # ================= T-TEST FAMILY =================

    def t_test_one_sample(self, column: str, mu: float) -> StatTestResult:
        series = self._engine.get_column(column)
        validate_numeric_column(series, column)
        clean = series.dropna().to_numpy(dtype=float)
        validate_min_sample_size(len(clean), 2, "One-sample t-test")
        validate_has_variance(clean, column, "One-sample t-test")

        t_stat, p_value = scipy_stats.ttest_1samp(clean, popmean=mu)
        sd = np.std(clean, ddof=1)
        d = (np.mean(clean) - mu) / sd if sd > 0 else 0.0

        return StatTestResult(
            statistic=float(t_stat), p_value=float(p_value), effect_size=float(d),
            effect_size_name="Cohen's d", df=float(len(clean) - 1), n=len(clean),
            data_quality=self._quality(len(clean)), test_name="One-Sample T-Test",
            groups={"sample_mean": float(np.mean(clean)), "test_value": float(mu), "sample_std": float(sd)},
        )

    def t_test_independent(self, value_col: str, group_col: str, group1, group2,
                            equal_var: bool = True) -> StatTestResult:
        value_series = self._engine.get_column(value_col)
        group_series = self._engine.get_column(group_col)
        validate_numeric_column(value_series, value_col)
        validate_grouping_column(group_series, group_col, min_groups=2)

        mask = ~(value_series.isna() | group_series.isna())
        v = value_series[mask].to_numpy(dtype=float)
        g = group_series[mask].to_numpy()

        v1 = v[g == group1]
        v2 = v[g == group2]
        if len(v1) < 2 or len(v2) < 2:
            raise ValueError(
                f"Each group needs at least 2 observations. Got {len(v1)} for "
                f"'{group1}' and {len(v2)} for '{group2}'."
            )
        validate_has_variance(v, value_col, "Independent t-test")

        t_stat, p_value = scipy_stats.ttest_ind(v1, v2, equal_var=equal_var)
        pooled_sd = np.sqrt(
            ((len(v1) - 1) * np.var(v1, ddof=1) + (len(v2) - 1) * np.var(v2, ddof=1))
            / (len(v1) + len(v2) - 2)
        )
        d = (np.mean(v1) - np.mean(v2)) / pooled_sd if pooled_sd > 0 else 0.0
        df_val = (len(v1) + len(v2) - 2) if equal_var else self._welch_df(v1, v2)

        return StatTestResult(
            statistic=float(t_stat), p_value=float(p_value), effect_size=float(d),
            effect_size_name="Cohen's d", df=float(df_val), n=len(v1) + len(v2),
            data_quality=self._quality(len(v1) + len(v2)),
            test_name="Independent T-Test" + (" (Welch)" if not equal_var else " (Student)"),
            groups={
                str(group1): {"mean": float(np.mean(v1)), "std": float(np.std(v1, ddof=1)), "n": int(len(v1))},
                str(group2): {"mean": float(np.mean(v2)), "std": float(np.std(v2, ddof=1)), "n": int(len(v2))},
            },
        )

    @staticmethod
    def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
        v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
        n1, n2 = len(a), len(b)
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        return float(num / den) if den > 0 else float(n1 + n2 - 2)

    def t_test_paired(self, col1: str, col2: str) -> StatTestResult:
        s1 = self._engine.get_column(col1)
        s2 = self._engine.get_column(col2)
        validate_numeric_column(s1, col1)
        validate_numeric_column(s2, col2)

        mask = ~(s1.isna() | s2.isna())
        a = s1[mask].to_numpy(dtype=float)
        b = s2[mask].to_numpy(dtype=float)
        validate_min_sample_size(len(a), 2, "Paired t-test")
        validate_has_variance(a - b, f"{col1} - {col2}", "Paired t-test")

        t_stat, p_value = scipy_stats.ttest_rel(a, b)
        diff = a - b
        sd_diff = np.std(diff, ddof=1)
        d = np.mean(diff) / sd_diff if sd_diff > 0 else 0.0

        return StatTestResult(
            statistic=float(t_stat), p_value=float(p_value), effect_size=float(d),
            effect_size_name="Cohen's d (z)", df=float(len(a) - 1), n=len(a),
            data_quality=self._quality(len(a)), test_name="Paired T-Test",
            groups={
                col1: {"mean": float(np.mean(a))}, col2: {"mean": float(np.mean(b))},
                "mean_difference": float(np.mean(diff)),
            },
        )

    # ================= ANOVA FAMILY =================

    def _run_anova_table(self, formula: str, typ: int = 2):
        df = self._engine.to_dataframe()
        try:
            model = smf.ols(formula, data=df).fit()
        except Exception as e:
            raise ValueError(f"ANOVA model failed to fit: {e}") from e
        aov = sm.stats.anova_lm(model, typ=typ)
        return model, aov

    @staticmethod
    def _terms_from_aov(aov, name_map=None) -> List[ANOVATerm]:
        name_map = name_map or {}
        resid_row = aov.loc["Residual"]
        resid_ss = float(resid_row["sum_sq"])
        terms = []
        for term_name in aov.index:
            if term_name == "Residual":
                continue
            row = aov.loc[term_name]
            denom = row["sum_sq"] + resid_ss
            partial_eta = float(row["sum_sq"] / denom) if denom > 0 else None
            clean_name = term_name
            for raw, pretty in name_map.items():
                clean_name = clean_name.replace(raw, pretty)
            terms.append(ANOVATerm(
                term=clean_name, sum_sq=float(row["sum_sq"]), df=float(row["df"]),
                f_value=float(row["F"]), p_value=float(row["PR(>F)"]), partial_eta_sq=partial_eta,
            ))
        return terms

    def anova_oneway(self, value_col: str, group_col: str, post_hoc: bool = True) -> ANOVAResult:
        validate_numeric_column(self._engine.get_column(value_col), value_col)
        validate_grouping_column(self._engine.get_column(group_col), group_col, min_groups=2)
        validate_has_variance(
            self._engine.get_column(value_col).dropna().to_numpy(dtype=float), value_col, "One-Way ANOVA"
        )

        formula = f"{value_col} ~ C({group_col})"
        model, aov = self._run_anova_table(formula, typ=2)
        n_used = int(model.nobs)
        terms = self._terms_from_aov(aov, {f"C({group_col})": group_col})
        resid_row = aov.loc["Residual"]

        df_clean = self._engine.to_dataframe()[[value_col, group_col]].dropna()
        group_summaries = [
            GroupSummary(group=str(g), n=len(sub), mean=float(sub[value_col].mean()),
                         std=float(sub[value_col].std(ddof=1)) if len(sub) > 1 else 0.0)
            for g, sub in df_clean.groupby(group_col)
        ]

        posthoc_results = None
        if post_hoc:
            posthoc_results = self._tukey_posthoc(
                df_clean[value_col].to_numpy(dtype=float), df_clean[group_col].astype(str).to_numpy()
            )

        return ANOVAResult(
            model_type="One-Way ANOVA", terms=terms,
            residual_sum_sq=float(resid_row["sum_sq"]), residual_df=float(resid_row["df"]),
            r_squared=float(model.rsquared), n=n_used, group_summaries=group_summaries,
            post_hoc=posthoc_results, dv_name=value_col, data_quality=self._quality(n_used),
        )

    def anova_twoway(self, value_col: str, factor1: str, factor2: str) -> ANOVAResult:
        for col in (factor1, factor2):
            validate_grouping_column(self._engine.get_column(col), col, min_groups=2)
        validate_numeric_column(self._engine.get_column(value_col), value_col)
        validate_has_variance(
            self._engine.get_column(value_col).dropna().to_numpy(dtype=float), value_col, "Two-Way ANOVA"
        )

        formula = f"{value_col} ~ C({factor1}) * C({factor2})"
        model, aov = self._run_anova_table(formula, typ=2)
        n_used = int(model.nobs)
        terms = self._terms_from_aov(aov, {f"C({factor1})": factor1, f"C({factor2})": factor2})
        resid_row = aov.loc["Residual"]

        df_clean = self._engine.to_dataframe()[[value_col, factor1, factor2]].dropna()
        group_summaries = [
            GroupSummary(group=f"{g1} / {g2}", n=len(sub), mean=float(sub[value_col].mean()),
                         std=float(sub[value_col].std(ddof=1)) if len(sub) > 1 else 0.0)
            for (g1, g2), sub in df_clean.groupby([factor1, factor2])
        ]

        return ANOVAResult(
            model_type=f"Two-Way ANOVA ({factor1} × {factor2})", terms=terms,
            residual_sum_sq=float(resid_row["sum_sq"]), residual_df=float(resid_row["df"]),
            r_squared=float(model.rsquared), n=n_used, group_summaries=group_summaries,
            dv_name=value_col, data_quality=self._quality(n_used),
        )

    def ancova(self, value_col: str, group_col: str, covariate_col: str) -> ANOVAResult:
        validate_grouping_column(self._engine.get_column(group_col), group_col, min_groups=2)
        validate_numeric_column(self._engine.get_column(value_col), value_col)
        validate_numeric_column(self._engine.get_column(covariate_col), covariate_col)
        validate_has_variance(
            self._engine.get_column(value_col).dropna().to_numpy(dtype=float), value_col, "ANCOVA"
        )

        formula = f"{value_col} ~ C({group_col}) + {covariate_col}"
        model, aov = self._run_anova_table(formula, typ=2)
        n_used = int(model.nobs)
        terms = self._terms_from_aov(aov, {f"C({group_col})": group_col})
        resid_row = aov.loc["Residual"]

        df_clean = self._engine.to_dataframe()[[value_col, group_col, covariate_col]].dropna()
        group_summaries = [
            GroupSummary(group=str(g), n=len(sub), mean=float(sub[value_col].mean()),
                         std=float(sub[value_col].std(ddof=1)) if len(sub) > 1 else 0.0)
            for g, sub in df_clean.groupby(group_col)
        ]

        return ANOVAResult(
            model_type=f"ANCOVA (covariate: {covariate_col})", terms=terms,
            residual_sum_sq=float(resid_row["sum_sq"]), residual_df=float(resid_row["df"]),
            r_squared=float(model.rsquared), n=n_used, group_summaries=group_summaries,
            dv_name=value_col, data_quality=self._quality(n_used),
        )

    def anova_repeated_measures(self, subject_col: str, within_col: str, value_col: str) -> ANOVAResult:
        df_clean = self._engine.to_dataframe()[[subject_col, within_col, value_col]].dropna()
        n_used = len(df_clean)
        validate_has_variance(df_clean[value_col].to_numpy(dtype=float), value_col, "Repeated Measures ANOVA")
        try:
            rm_fit = AnovaRM(df_clean, depvar=value_col, subject=subject_col, within=[within_col]).fit()
        except Exception as e:
            raise ValueError(
                f"Repeated measures ANOVA failed: {e}. This usually means the design is "
                f"unbalanced — every subject needs exactly one observation per level of "
                f"'{within_col}'."
            ) from e

        row = rm_fit.anova_table.loc[within_col]
        term = ANOVATerm(
            term=within_col, sum_sq=None, df=float(row["Num DF"]), df2=float(row["Den DF"]),
            f_value=float(row["F Value"]), p_value=float(row["Pr > F"]), partial_eta_sq=None,
        )

        group_summaries = [
            GroupSummary(group=str(g), n=len(sub), mean=float(sub[value_col].mean()),
                         std=float(sub[value_col].std(ddof=1)) if len(sub) > 1 else 0.0)
            for g, sub in df_clean.groupby(within_col)
        ]

        return ANOVAResult(
            model_type="Repeated Measures ANOVA", terms=[term], n=n_used,
            group_summaries=group_summaries, dv_name=value_col, data_quality=self._quality(n_used),
        )

    @staticmethod
    def _tukey_posthoc(values: np.ndarray, groups: np.ndarray) -> List[PostHocResult]:
        res = pairwise_tukeyhsd(values, groups)
        table = res.summary().data
        header = table[0]
        idx = {name: i for i, name in enumerate(header)}
        results = []
        for row in table[1:]:
            results.append(PostHocResult(
                group1=str(row[idx["group1"]]), group2=str(row[idx["group2"]]),
                mean_diff=float(row[idx["meandiff"]]), p_adj=float(row[idx["p-adj"]]),
                ci_lower=float(row[idx["lower"]]), ci_upper=float(row[idx["upper"]]),
                reject=bool(row[idx["reject"]]),
            ))
        return results

    # ================= MANOVA =================

    def manova(self, dv_cols: List[str], group_col: str) -> MANOVAResult:
        from statsmodels.multivariate.manova import MANOVA

        if len(dv_cols) < 2:
            raise ValueError("MANOVA requires at least 2 dependent variables.")
        for c in dv_cols:
            validate_numeric_column(self._engine.get_column(c), c)
        validate_grouping_column(self._engine.get_column(group_col), group_col, min_groups=2)

        df_clean = self._engine.to_dataframe()[dv_cols + [group_col]].dropna()
        n_used = len(df_clean)
        formula = f"{' + '.join(dv_cols)} ~ C({group_col})"

        try:
            fit = MANOVA.from_formula(formula, data=df_clean)
            mv = fit.mv_test()
        except Exception as e:
            raise ValueError(f"MANOVA failed to fit: {e}") from e

        term_key = f"C({group_col})"
        if term_key not in mv.results:
            candidates = [k for k in mv.results.keys() if "Intercept" not in k]
            if not candidates:
                raise ValueError("MANOVA produced no usable group effect term.")
            term_key = candidates[0]

        stat_df = mv.results[term_key]["stat"]
        stats_list = [
            MANOVAStatistic(
                name=str(name), value=float(stat_df.loc[name, "Value"]),
                f_value=float(stat_df.loc[name, "F Value"]), df1=float(stat_df.loc[name, "Num DF"]),
                df2=float(stat_df.loc[name, "Den DF"]), p_value=float(stat_df.loc[name, "Pr > F"]),
            )
            for name in stat_df.index
        ]

        return MANOVAResult(
            term=group_col, statistics=stats_list, dv_names=dv_cols,
            group_col=group_col, n=n_used,
        )
