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


class SPSSMirror:
    """
    Central facade. Statistics engines attach themselves here phase by
    phase (descriptive, regression, frequentist, etc.).
    """

    def __init__(self):
        self._engine = DataEngine()
        self._descriptive_engine = DescriptiveEngine(self._engine)
        self._regression_engine = RegressionEngine(self._engine)
        self._frequentist_engine = FrequentistParametricEngine(self._engine)
        self._nonparametric_engine = FrequentistNonparametricEngine(self._engine)
        self._categorical_engine = CategoricalEngine(self._engine)
        self._correlation_engine = CorrelationEngine(self._engine)
        self._psychometrics_engine = PsychometricsEngine(self._engine)
        self._effect_size_engine = EffectSizeEngine(self._engine)
        self._power_engine = PowerAnalysisEngine()
        self._diagnostics_engine = DiagnosticsEngine(self._engine)
        self._mixed_models_engine = MixedModelsEngine(self._engine)
        self._bayesian_engine = BayesianEngine(self._engine)
        self._timeseries_engine = TimeSeriesEngine(self._engine)
        self._survival_engine = SurvivalEngine(self._engine)
        self._multivariate_engine = MultivariateEngine(self._engine)

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
        changes here will NOT affect this SPSSMirror instance. (Internal
        engines use DataEngine.to_dataframe() directly without copying,
        since library code never mutates what it reads; this public-facing
        copy exists because external caller code should not be expected to
        follow that same discipline.)"""
        return self._engine.to_dataframe().copy()

    def formula(self, formula_str: str) -> ParsedFormula:
        """Low-level formula parse, exposed for engines built in later phases."""
        return parse_formula(formula_str, self._engine.to_dataframe())

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
        # internal accessor used by statistics engines once they're attached
        return self._engine
