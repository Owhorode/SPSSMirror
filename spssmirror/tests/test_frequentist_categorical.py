import numpy as np
import pytest
from spssmirror import SPSSMirror


class TestFrequentistParametric:
    def test_independent_ttest_detects_known_difference(self, two_group_data):
        r = two_group_data.frequentist().t_test_independent("score", "grp", "A", "B")
        assert r.p_value < 0.01

    def test_paired_ttest_detects_known_shift(self, rng):
        before = rng.normal(70, 10, 30)
        after = before + rng.normal(5, 3, 30)
        r = SPSSMirror().load_dict({"before": before.tolist(), "after": after.tolist()}) \
            .frequentist().t_test_paired("before", "after")
        assert r.p_value < 0.001

    def test_anova_oneway_posthoc_distinguishes_true_from_false_diff(self, three_group_data):
        r = three_group_data.frequentist().anova_oneway("score", "grp", post_hoc=True)
        ab = [p for p in r.post_hoc if {p.group1, p.group2} == {"A", "B"}][0]
        ac = [p for p in r.post_hoc if {p.group1, p.group2} == {"A", "C"}][0]
        assert ab.reject is True
        assert ac.reject is False

    def test_anova_twoway_detects_engineered_interaction(self, rng):
        import pandas as pd
        n = 30
        rows = []
        for f1 in ["Low", "High"]:
            for f2 in ["X", "Y"]:
                base = 50 + (5 if f1 == "High" else 0) + (5 if f2 == "Y" else 0) + (10 if f1 == "High" and f2 == "Y" else 0)
                rows += [(v, f1, f2) for v in rng.normal(base, 4, n)]
        df = pd.DataFrame(rows, columns=["score", "f1", "f2"])
        r = SPSSMirror().load_dataframe(df).frequentist().anova_twoway("score", "f1", "f2")
        interaction = [t for t in r.terms if ":" in t.term][0]
        assert interaction.p_value < 0.01

    def test_single_group_anova_raises(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"y": [1, 2, 3], "g": ["A", "A", "A"]}).frequentist().anova_oneway("y", "g")


class TestFrequentistNonparametric:
    def test_mann_whitney_detects_known_difference(self, two_group_data):
        r = two_group_data.nonparametric().mann_whitney_u("score", "grp", "A", "B")
        assert r.p_value < 0.01

    def test_wilcoxon_rejects_all_zero_differences(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"a": [1, 2, 3], "b": [1, 2, 3]}).nonparametric().wilcoxon_signed_rank("a", "b")

    def test_kruskal_wallis_detects_known_difference(self, three_group_data):
        r = three_group_data.nonparametric().kruskal_wallis("score", "grp")
        assert r.p_value < 0.001

    def test_friedman_rejects_unbalanced_design(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({
                "subj": [1, 1, 2], "cond": ["T1", "T2", "T1"], "val": [1, 2, 3],
            }).nonparametric().friedman_test("subj", "cond", "val")


class TestCategorical:
    def test_chi_square_detects_engineered_association(self, rng):
        grp = rng.choice(["A", "B"], 300)
        outcome = np.where(grp == "A", rng.choice(["Yes", "No"], 300, p=[0.2, 0.8]),
                            rng.choice(["Yes", "No"], 300, p=[0.8, 0.2]))
        r = SPSSMirror().load_dict({"grp": grp.tolist(), "outcome": outcome.tolist()}).categorical().chi_square_independence("grp", "outcome")
        assert r.chi_square_p < 0.001
        assert r.cramers_v is not None

    def test_fishers_exact_requires_2x2(self):
        with pytest.raises(ValueError):
            SPSSMirror().load_dict({"a": ["X", "Y", "Z"], "b": ["P", "Q", "P"]}).categorical().fishers_exact("a", "b")
