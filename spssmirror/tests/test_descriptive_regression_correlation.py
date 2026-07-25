import numpy as np
import pytest
from spssmirror import SPSSMirror


class TestDescriptive:
    def test_summary_recovers_known_mean(self, rng):
        data = rng.normal(100, 15, 300)
        r = SPSSMirror().load_dict({"x": data.tolist()}).descriptive().summary("x")
        assert abs(r.mean - 100) < 3
        assert r.n == 300 and r.n_missing == 0

    def test_summary_tracks_missing(self):
        vals = [1.0, 2.0, None, 4.0, None]
        r = SPSSMirror().load_dict({"x": vals}).descriptive().summary("x")
        assert r.n == 3 and r.n_missing == 2

    def test_summary_rejects_non_numeric(self):
        with pytest.raises(TypeError):
            SPSSMirror().load_dict({"x": ["a", "b", "c"]}).descriptive().summary("x")

    def test_frequency_table_counts_sum_to_n(self, rng):
        vals = rng.choice([1, 2, 3, 4], size=150).tolist()
        r = SPSSMirror().load_dict({"q": vals}).descriptive().frequency_table("q")
        assert sum(r.frequencies) == 150

    def test_crosstab_detects_engineered_association(self, rng):
        grp = rng.choice(["A", "B"], 300)
        outcome = np.where(grp == "A", rng.choice(["Yes", "No"], 300, p=[0.2, 0.8]),
                            rng.choice(["Yes", "No"], 300, p=[0.8, 0.2]))
        r = SPSSMirror().load_dict({"grp": grp.tolist(), "outcome": outcome.tolist()}).descriptive().crosstab("grp", "outcome")
        assert r.chi_square_p < 0.001


class TestRegression:
    def test_linear_recovers_known_coefficients(self, linear_regression_data):
        r = linear_regression_data.regression().linear("y ~ x1 + x2")
        assert abs(r.coefficients[1].b - 3.0) < 0.2
        assert abs(r.coefficients[2].b - (-1.5)) < 0.2
        assert r.r_squared > 0.9

    def test_logistic_rejects_non_binary(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"y": [1, 2, 3, 1, 2], "x": [1, 2, 3, 4, 5]}).regression().logistic("y ~ x")

    def test_robust_outperforms_ols_with_outliers(self, rng):
        n = 100
        x = rng.normal(0, 1, n)
        y = 5 + 2 * x + rng.normal(0, 0.3, n)
        y[:5] += 50
        mirror = SPSSMirror().load_dict({"y": y.tolist(), "x": x.tolist()})
        ols = mirror.regression().linear("y ~ x")
        robust = mirror.regression().robust("y ~ x")
        assert abs(robust.coefficients[1].b - 2.0) < abs(ols.coefficients[1].b - 2.0)

    def test_regularized_models_omit_fabricated_inference(self, linear_regression_data):
        r = linear_regression_data.regression().ridge("y ~ x1 + x2", alpha=1.0)
        assert r.coefficients[1].std_error is None
        assert r.coefficients[1].p_value is None

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"y": [1], "x": [1]}).regression().linear("y ~ x")


class TestCorrelations:
    def test_pearson_matches_known_signal(self, correlated_pair_data):
        r = correlated_pair_data.correlations().pearson("x", "y")
        assert r.coefficient > 0.7
        assert r.p_value < 0.001

    def test_partial_correlation_removes_confound(self, rng):
        n = 200
        confound = rng.normal(0, 1, n)
        x = confound + rng.normal(0, 0.3, n)
        y = confound + rng.normal(0, 0.3, n)
        mirror = SPSSMirror().load_dict({"x": x.tolist(), "y": y.tolist(), "z": confound.tolist()})
        zero_order = mirror.correlations().pearson("x", "y")
        partial = mirror.correlations().partial("x", "y", covariates=["z"])
        assert abs(partial.coefficient) < abs(zero_order.coefficient) - 0.3

    def test_point_biserial_rejects_non_binary(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"b": [1, 2, 3], "c": [1, 2, 3]}).correlations().point_biserial("b", "c")

    def test_correlation_matrix_diagonal_is_one(self, correlated_pair_data):
        mat = correlated_pair_data.correlations().correlation_matrix(["x", "y"])
        assert abs(mat.matrix[0][0] - 1.0) < 1e-9
        assert abs(mat.matrix[1][1] - 1.0) < 1e-9
