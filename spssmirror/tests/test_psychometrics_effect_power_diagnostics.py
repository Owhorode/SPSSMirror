import numpy as np
import pytest
from spssmirror import SPSSMirror


class TestPsychometrics:
    def test_cronbach_alpha_high_on_real_signal(self, reliable_scale_data):
        mirror, items = reliable_scale_data
        r = mirror.psychometrics().cronbach_alpha(list(items.keys()))
        assert r.statistic > 0.8

    def test_cronbach_alpha_low_on_noise(self, rng):
        random_items = {f"r{i}": rng.normal(0, 1, 300).tolist() for i in range(6)}
        r = SPSSMirror().load_dict(random_items).psychometrics().cronbach_alpha(list(random_items.keys()))
        assert r.statistic < 0.3

    def test_efa_recovers_known_two_factor_structure(self, two_factor_data):
        mirror, items = two_factor_data
        fa = mirror.psychometrics().efa(items, n_factors=2, rotation="varimax")
        assert fa.n_factors == 2
        a_loadings = fa.loadings[:3]
        b_loadings = fa.loadings[3:]
        for l in a_loadings:
            assert max(abs(l[0]), abs(l[1])) > 0.5
        for l in b_loadings:
            assert max(abs(l[0]), abs(l[1])) > 0.5

    def test_single_item_alpha_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"q1": [1, 2, 3]}).psychometrics().cronbach_alpha(["q1"])


class TestEffectSizes:
    def test_cohens_d_matches_ttest_embedded_value(self, two_group_data):
        d = two_group_data.effect_sizes().cohens_d("score", "grp", "A", "B")
        t = two_group_data.frequentist().t_test_independent("score", "grp", "A", "B")
        assert abs(d.value - t.effect_size) < 1e-9

    def test_hedges_g_smaller_magnitude_than_d(self, two_group_data):
        d = two_group_data.effect_sizes().cohens_d("score", "grp", "A", "B")
        g = two_group_data.effect_sizes().hedges_g("score", "grp", "A", "B")
        assert abs(g.value) < abs(d.value)

    def test_odds_ratio_reflects_strong_association(self):
        mirror = SPSSMirror().load_dict({
            "exposure": ["Yes"] * 80 + ["No"] * 80,
            "outcome": ["Disease"] * 60 + ["Healthy"] * 20 + ["Disease"] * 20 + ["Healthy"] * 60,
        })
        r = mirror.effect_sizes().odds_ratio("exposure", "outcome")
        assert r.value > 5 or r.value < 0.2
        assert not (r.ci_lower < 1 < r.ci_upper)


class TestPowerAnalysis:
    def test_ttest_power_matches_cohen_1988_benchmark(self):
        r = SPSSMirror().power().power_ttest_independent(effect_size=0.5, alpha=0.05, power=0.80)
        assert 60 < r.n < 70

    def test_anova_power_matches_cohen_1988_benchmark(self):
        r = SPSSMirror().power().power_anova(effect_size=0.25, k_groups=3, alpha=0.05, power=0.80)
        assert 45 < r.n < 60

    def test_overspecified_call_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().power().power_ttest_independent(effect_size=0.5, n=64, power=0.8, alpha=0.05)


class TestDiagnostics:
    def test_normality_accepts_normal_data(self, rng):
        r = SPSSMirror().load_dict({"x": rng.normal(50, 10, 200).tolist()}).diagnostics().normality_tests("x")
        assert r.is_normal is True

    def test_normality_rejects_skewed_data(self, rng):
        r = SPSSMirror().load_dict({"x": rng.exponential(2, 200).tolist()}).diagnostics().normality_tests("x")
        assert r.is_normal is False

    def test_vif_flags_collinear_predictors(self, rng):
        x1 = rng.normal(0, 1, 200)
        x2 = x1 + rng.normal(0, 0.02, 200)
        x3 = rng.normal(0, 1, 200)
        mirror = SPSSMirror().load_dict({"x1": x1.tolist(), "x2": x2.tolist(), "x3": x3.tolist()})
        r = mirror.diagnostics().vif(["x1", "x2", "x3"])
        assert r.items[0].vif > 10
        assert r.items[2].vif < 5

    def test_residual_diagnostics_flags_injected_influential_point(self, rng):
        n = 50
        x = rng.normal(0, 1, n)
        y = 3 + 2 * x + rng.normal(0, 0.5, n)
        x[0], y[0] = 10.0, -20.0
        mirror = SPSSMirror().load_dict({"y": y.tolist(), "x": x.tolist()})
        r = mirror.diagnostics().residual_diagnostics("y ~ x")
        assert 0 in r.influential_indices
