import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from spssmirror.core._engine import DataEngine
from spssmirror.statistics.descriptive import DescriptiveEngine
from spssmirror.models._results import StatTestResult, DataQuality, CrossTabResult


class CategoricalEngine:
    def __init__(self, engine: DataEngine):
        self._engine = engine
        self._descriptive = DescriptiveEngine(engine)

    def _quality(self, n_used: int) -> DataQuality:
        n_original = self._engine.shape()[0]
        return DataQuality(
            n_rows_original=n_original, n_rows_analyzed=n_used,
            n_nulls_dropped=n_original - n_used,
            max_missing_ratio=(n_original - n_used) / n_original if n_original > 0 else 0.0,
        )

    def chi_square_independence(self, col1: str, col2: str) -> CrossTabResult:
        """Delegates to the crosstab engine — same table, same chi-square,
        now also carrying Cramér's V as an effect size."""
        result = self._descriptive.crosstab(col1, col2, with_chi_square=True)
        if result.chi_square is None:
            raise ValueError(
                f"Chi-square could not be computed for '{col1}' × '{col2}' "
                f"(degenerate table — one dimension has fewer than 2 categories)."
            )
        return result

    def fishers_exact(self, col1: str, col2: str) -> StatTestResult:
        s1, s2 = self._engine.get_column(col1), self._engine.get_column(col2)
        mask = ~(s1.isna() | s2.isna())
        table = pd.crosstab(s1[mask], s2[mask])

        if table.shape != (2, 2):
            raise ValueError(
                f"Fisher's exact test requires a 2x2 table. Got {table.shape[0]}x{table.shape[1]} "
                f"for '{col1}' × '{col2}'. Use chi_square_independence() for larger tables."
            )

        odds_ratio, p_value = scipy_stats.fisher_exact(table.to_numpy())
        n = int(table.to_numpy().sum())

        return StatTestResult(
            statistic=float(odds_ratio), p_value=float(p_value), effect_size=float(odds_ratio),
            effect_size_name="Odds Ratio", n=n, data_quality=self._quality(n),
            test_name="Fisher's Exact Test",
        )

    def mcnemar_test(self, col1: str, col2: str, exact: bool = True) -> StatTestResult:
        s1, s2 = self._engine.get_column(col1), self._engine.get_column(col2)
        mask = ~(s1.isna() | s2.isna())
        table = pd.crosstab(s1[mask], s2[mask])

        if table.shape != (2, 2):
            raise ValueError(
                f"McNemar's test requires a 2x2 table of matched-pair categories. "
                f"Got {table.shape[0]}x{table.shape[1]} for '{col1}' × '{col2}'."
            )

        result = sm_mcnemar(table.to_numpy(), exact=exact)
        n = int(table.to_numpy().sum())

        return StatTestResult(
            statistic=float(result.statistic), p_value=float(result.pvalue),
            n=n, data_quality=self._quality(n), test_name="McNemar's Test",
        )
