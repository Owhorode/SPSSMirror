from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List


class DataQuality(BaseModel):
    model_config = ConfigDict(frozen=True)
    n_rows_original: int
    n_rows_analyzed: int
    n_nulls_dropped: int
    max_missing_ratio: float


class BaseResult(BaseModel):
    """Common ancestor for every SPSSMirror result."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class StatTestResult(BaseResult):
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    df: Optional[float] = None
    n: int
    data_quality: DataQuality
    test_name: str
    groups: Optional[Dict[str, Any]] = None


class CorrelationResult(BaseResult):
    coefficient: float
    p_value: float
    n: int
    method: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    data_quality: DataQuality
    x_name: Optional[str] = None
    y_name: Optional[str] = None
    x_values: List[float] = Field(default_factory=list, repr=False)
    y_values: List[float] = Field(default_factory=list, repr=False)


class CorrelationMatrixResult(BaseResult):
    variables: List[str]
    matrix: List[List[float]]
    p_matrix: List[List[float]]
    method: str
    n: int


class RegressionCoefficient(BaseModel):
    model_config = ConfigDict(frozen=True)
    term: str
    b: float
    std_error: Optional[float] = None
    t_value: Optional[float] = None
    z_value: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    beta: Optional[float] = None  # standardized coefficient


class RegressionResult(BaseResult):
    model_type: str
    r_squared: Optional[float] = None
    adj_r_squared: Optional[float] = None
    pseudo_r_squared: Optional[float] = None
    f_statistic: Optional[float] = None
    f_p_value: Optional[float] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    n: int
    df_model: int
    df_residual: int
    coefficients: List[RegressionCoefficient]
    residuals: List[float] = Field(default_factory=list, repr=False)
    fitted_values: List[float] = Field(default_factory=list, repr=False)
    y_name: str
    data_quality: DataQuality


class PsychometricResult(BaseResult):
    statistic: float
    n_items: int
    n_respondents: int
    threshold: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    data_quality: DataQuality
    metric_name: str


class FrequencyTableResult(BaseResult):
    column: str
    categories: List[str]
    frequencies: List[int]
    percentages: List[float]
    cumulative_percentages: List[float]
    n: int
    n_missing: int


class ANOVATerm(BaseModel):
    model_config = ConfigDict(frozen=True)
    term: str
    sum_sq: Optional[float] = None
    df: float
    df2: Optional[float] = None
    f_value: Optional[float] = None
    p_value: Optional[float] = None
    partial_eta_sq: Optional[float] = None


class PostHocResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    group1: str
    group2: str
    mean_diff: float
    p_adj: float
    ci_lower: float
    ci_upper: float
    reject: bool


class GroupSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    group: str
    n: int
    mean: float
    std: float


class ANOVAResult(BaseResult):
    model_type: str
    terms: List[ANOVATerm]
    residual_sum_sq: Optional[float] = None
    residual_df: Optional[float] = None
    r_squared: Optional[float] = None
    n: int
    group_summaries: List[GroupSummary] = Field(default_factory=list)
    post_hoc: Optional[List[PostHocResult]] = None
    dv_name: str
    data_quality: DataQuality


class MANOVAStatistic(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    value: float
    f_value: Optional[float] = None
    df1: Optional[float] = None
    df2: Optional[float] = None
    p_value: Optional[float] = None


class MANOVAResult(BaseResult):
    term: str
    statistics: List[MANOVAStatistic]
    dv_names: List[str]
    group_col: str
    n: int


class ItemStat(BaseModel):
    model_config = ConfigDict(frozen=True)
    item: str
    mean: float
    std: float
    item_total_corr: float
    alpha_if_deleted: float


class ItemAnalysisResult(BaseResult):
    items: List[ItemStat]
    overall_alpha: float
    n: int
    n_items: int


class FactorAnalysisResult(BaseResult):
    items: List[str]
    n_factors: int
    loadings: List[List[float]]  # item x factor
    eigenvalues: List[float]
    variance_explained: List[float]
    cumulative_variance: List[float]
    communalities: List[float]
    rotation: str
    n: int


class EffectSizeResult(BaseResult):
    name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    interpretation: str
    n: int
    details: Dict[str, Any] = Field(default_factory=dict)


class PowerResult(BaseResult):
    test_type: str
    effect_size: float
    alpha: float
    power: Optional[float] = None
    n: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PowerCurveResult(BaseResult):
    test_type: str
    x_label: str
    x_values: List[float]
    y_values: List[float]
    target_power: float = 0.8
    context: Dict[str, Any] = Field(default_factory=dict)


class NormalityTestStat(BaseModel):
    model_config = ConfigDict(frozen=True)
    test_name: str
    statistic: float
    p_value: Optional[float] = None
    note: Optional[str] = None


class NormalityResult(BaseResult):
    column: str
    tests: List[NormalityTestStat]
    n: int
    mean: float
    std: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    is_normal: bool
    values: List[float] = Field(default_factory=list, repr=False)


class VIFItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    predictor: str
    vif: float
    tolerance: float


class VIFResult(BaseResult):
    items: List[VIFItem]
    n: int
    concerning_threshold: float = 10.0


class ResidualDiagnosticsResult(BaseResult):
    fitted: List[float] = Field(default_factory=list, repr=False)
    residuals: List[float] = Field(default_factory=list, repr=False)
    standardized_residuals: List[float] = Field(default_factory=list, repr=False)
    leverage: List[float] = Field(default_factory=list, repr=False)
    cooks_distance: List[float] = Field(default_factory=list, repr=False)
    influential_indices: List[int]
    n: int
    k_predictors: int
    y_name: str


class OutlierResult(BaseResult):
    column: str
    method: str
    lower_bound: float
    upper_bound: float
    outlier_indices: List[int]
    outlier_values: List[float]
    n: int
    n_outliers: int
    values: List[float] = Field(default_factory=list, repr=False)


class RandomEffectVarianceComponent(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    variance: float


class GroupRandomEffect(BaseModel):
    model_config = ConfigDict(frozen=True)
    group: str
    effects: Dict[str, float]


class MixedModelResult(BaseResult):
    fixed_effects: List[RegressionCoefficient]
    variance_components: List[RandomEffectVarianceComponent]
    group_effects: List[GroupRandomEffect]
    icc: Optional[float] = None
    group_col: str
    n_groups: int
    n: int
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    estimation_method: str
    y_name: str
    converged: bool
    data_quality: DataQuality


class PosteriorParameter(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    mean: float
    std: float
    hdi_lower: float
    hdi_upper: float
    r_hat: Optional[float] = None
    ess_bulk: Optional[float] = None
    samples: List[float] = Field(default_factory=list, repr=False)


class BayesianTestResult(BaseResult):
    test_name: str
    parameters: List[PosteriorParameter]
    prob_direction: Optional[float] = None
    prob_direction_label: Optional[str] = None
    bayes_factor_10: Optional[float] = None
    hdi_prob: float
    n: int
    n_draws: int
    n_chains: int
    converged: bool
    data_quality: DataQuality


class BayesianRegressionResult(BaseResult):
    parameters: List[PosteriorParameter]
    hdi_prob: float
    n: int
    n_draws: int
    n_chains: int
    converged: bool
    y_name: str
    data_quality: DataQuality


class ARIMACoefficient(BaseModel):
    model_config = ConfigDict(frozen=True)
    term: str
    value: float
    std_error: Optional[float] = None
    p_value: Optional[float] = None


class ForecastPoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    step: int
    mean: float
    ci_lower: float
    ci_upper: float


class ARIMAResult(BaseResult):
    order: List[int]
    seasonal_order: Optional[List[int]] = None
    coefficients: List[ARIMACoefficient]
    aic: float
    bic: float
    log_likelihood: float
    fitted_values: List[float] = Field(default_factory=list, repr=False)
    residuals: List[float] = Field(default_factory=list, repr=False)
    original_values: List[float] = Field(default_factory=list, repr=False)
    forecast: List[ForecastPoint] = Field(default_factory=list)
    n: int
    y_name: str


class ExponentialSmoothingResult(BaseResult):
    trend: Optional[str] = None
    seasonal: Optional[str] = None
    seasonal_periods: Optional[int] = None
    smoothing_level: Optional[float] = None
    smoothing_trend: Optional[float] = None
    smoothing_seasonal: Optional[float] = None
    aic: Optional[float] = None
    fitted_values: List[float] = Field(default_factory=list, repr=False)
    original_values: List[float] = Field(default_factory=list, repr=False)
    forecast: List[ForecastPoint] = Field(default_factory=list)
    n: int
    y_name: str


class GARCHResult(BaseResult):
    omega: float
    alpha: List[float]
    beta: List[float]
    persistence: float
    aic: float
    bic: float
    conditional_volatility: List[float] = Field(default_factory=list, repr=False)
    original_values: List[float] = Field(default_factory=list, repr=False)
    forecast_variance: List[float] = Field(default_factory=list)
    n: int
    y_name: str


class ACFResult(BaseResult):
    acf_values: List[float]
    pacf_values: List[float]
    acf_confint: List[List[float]]
    pacf_confint: List[List[float]]
    n_lags: int
    n: int
    column: str


class StationarityResult(BaseResult):
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]
    n: int
    column: str


class SurvivalPoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    time: float
    survival_prob: float
    ci_lower: float
    ci_upper: float
    n_at_risk: int
    n_events: int


class KaplanMeierResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    group: Optional[str]
    curve: List[SurvivalPoint]
    median_survival: Optional[float] = None
    n: int
    n_events: int


class LogRankResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    statistic: float
    p_value: float
    groups_compared: List[str]
    n: int


class CoxCoefficient(BaseModel):
    model_config = ConfigDict(frozen=True)
    covariate: str
    coef: float
    hazard_ratio: float
    std_error: float
    z_value: float
    p_value: float
    ci_lower: float
    ci_upper: float
    hr_ci_lower: float
    hr_ci_upper: float


class CoxPHResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    coefficients: List[CoxCoefficient]
    concordance: float
    log_likelihood: float
    aic: float
    n: int
    n_events: int
    baseline_hazard_available: bool


class ParametricSurvivalResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    distribution: str
    coefficients: List[CoxCoefficient]
    log_likelihood: float
    aic: float
    n: int
    n_events: int


class PCAComponent(BaseModel):
    model_config = ConfigDict(frozen=True)
    component: str
    explained_variance: float
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    loadings: Dict[str, float]


class PCAResult(BaseResult):
    variables: List[str]
    components: List[PCAComponent]
    scores: List[List[float]] = Field(default_factory=list, repr=False)
    n: int


class ClusterProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    cluster: str
    n: int
    centroid: Dict[str, float]


class ClusteringResult(BaseResult):
    method: str
    variables: List[str]
    labels: List[int] = Field(default_factory=list, repr=False)
    profiles: List[ClusterProfile]
    n_clusters: int
    inertia: Optional[float] = None
    silhouette: Optional[float] = None
    linkage_matrix: List[List[float]] = Field(default_factory=list, repr=False)
    n: int


class DiscriminantFunctionCoefficient(BaseModel):
    model_config = ConfigDict(frozen=True)
    variable: str
    coefficients: Dict[str, float]


class DiscriminantResult(BaseResult):
    method: str
    group_col: str
    predictors: List[str]
    groups: List[str]
    accuracy: float
    coefficients: List[DiscriminantFunctionCoefficient]
    explained_variance_ratio: List[float] = Field(default_factory=list)
    confusion_matrix: List[List[int]]
    n: int


class CanonicalCorrelationResult(BaseResult):
    set1_variables: List[str]
    set2_variables: List[str]
    canonical_correlations: List[float]
    n_pairs: int
    n: int


class CrossTabResult(BaseResult):
    row_var: str
    col_var: str
    row_categories: List[str]
    col_categories: List[str]
    counts: List[List[int]]
    row_percentages: List[List[float]]
    chi_square: Optional[float] = None
    chi_square_p: Optional[float] = None
    cramers_v: Optional[float] = None
    n: int


class DescriptiveResult(BaseResult):
    column: str
    n: int
    n_missing: int
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    range_: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    values: List[float] = Field(default_factory=list, repr=False)
