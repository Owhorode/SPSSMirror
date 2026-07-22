import numpy as np
import pytest
from spssmirror import SPSSMirror


class TestMixedModels:
    def test_random_intercept_recovers_known_effects(self, rng):
        n_subjects, n_per = 30, 5
        subj = np.repeat(np.arange(n_subjects), n_per).tolist()
        time = np.tile(np.arange(n_per), n_subjects).tolist()
        subj_intercept = np.repeat(rng.normal(0, 3, n_subjects), n_per)
        y = (10 + 2 * np.array(time) + subj_intercept + rng.normal(0, 1, n_subjects * n_per)).tolist()
        mirror = SPSSMirror().load_dict({"subj": subj, "time": time, "y": y})
        r = mirror.mixed_models().linear_mixed_model("y ~ time", group_col="subj")
        assert abs(r.fixed_effects[1].b - 2) < 0.3
        assert r.icc > 0.5

    def test_reml_omits_aic_bic(self, rng):
        n_subjects, n_per = 20, 4
        subj = np.repeat(np.arange(n_subjects), n_per).tolist()
        subj_intercept = np.repeat(rng.normal(0, 2, n_subjects), n_per)
        y = (5 + subj_intercept + rng.normal(0, 1, n_subjects * n_per)).tolist()
        r = SPSSMirror().load_dict({"subj": subj, "y": y}).mixed_models().linear_mixed_model("y ~ 1", group_col="subj")
        assert r.aic is None and r.bic is None

    def test_single_group_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"y": list(range(8)), "g": ["A"] * 8}).mixed_models().linear_mixed_model("y ~ 1", group_col="g")


class TestBayesian:
    def test_bayesian_ttest_detects_known_difference(self, two_group_data):
        r = two_group_data.bayesian().bayesian_ttest("score", "grp", "A", "B", draws=500, tune=500, seed=0)
        diff = [p for p in r.parameters if p.name == "diff"][0]
        assert diff.hdi_upper < 0
        assert r.converged is True

    def test_proportion_test_is_closed_form_and_fast(self):
        import time
        t0 = time.time()
        r = SPSSMirror().bayesian().bayesian_proportion_test(successes=70, trials=100)
        assert time.time() - t0 < 1.0
        assert abs(r.parameters[0].mean - 0.7) < 0.05

    def test_proportion_test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            SPSSMirror().bayesian().bayesian_proportion_test(successes=150, trials=100)


class TestTimeSeries:
    def test_arima_recovers_known_ar_coefficient(self, ar1_series):
        r = ar1_series.timeseries().arima("y", order=(1, 0, 0))
        ar_coef = [c for c in r.coefficients if "ar.L1" in c.term][0]
        assert abs(ar_coef.value - 0.7) < 0.15

    def test_stationarity_distinguishes_ar1_from_random_walk(self, ar1_series, rng):
        stationary = ar1_series.timeseries().stationarity_test("y")
        assert stationary.is_stationary is True

        rw = np.cumsum(rng.normal(0, 1, 200))
        nonstationary = SPSSMirror().load_dict({"rw": rw.tolist()}).timeseries().stationarity_test("rw")
        assert nonstationary.is_stationary is False

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"y": [1, 2, 3, 4, 5]}).timeseries().arima("y", order=(1, 0, 0))


class TestSurvival:
    def test_kaplan_meier_group_a_outlives_group_b(self, survival_two_group_data):
        km = survival_two_group_data.survival().kaplan_meier("duration", "event", group_col="grp")
        med_a = [k.median_survival for k in km if k.group == "A"][0]
        med_b = [k.median_survival for k in km if k.group == "B"][0]
        assert med_a > med_b

    def test_logrank_detects_known_difference(self, survival_two_group_data):
        r = survival_two_group_data.survival().logrank_test("duration", "event", "grp")
        assert r.p_value < 0.001

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"d": [-1, 2, 3, 4, 5, 6, 7], "e": [1, 1, 0, 1, 1, 0, 1]}).survival().kaplan_meier("d", "e")


class TestMultivariate:
    def test_pca_captures_known_two_factor_structure(self, two_factor_data):
        mirror, items = two_factor_data
        r = mirror.multivariate().pca(items, n_components=2)
        assert r.components[1].cumulative_variance_ratio > 0.75

    def test_kmeans_perfectly_recovers_separated_clusters(self, separable_clusters_data):
        from itertools import permutations
        r = separable_clusters_data.multivariate().kmeans_clustering(["x", "y"], n_clusters=3, standardize=False)
        assert r.silhouette > 0.7

    def test_canonical_correlation_rejects_overlapping_sets(self, rng):
        mirror = SPSSMirror().load_dict({
            "x1": rng.normal(0, 1, 50).tolist(), "y2": rng.normal(0, 1, 50).tolist(),
        })
        with pytest.raises(ValueError):
            mirror.multivariate().canonical_correlation(["x1"], ["x1"])
