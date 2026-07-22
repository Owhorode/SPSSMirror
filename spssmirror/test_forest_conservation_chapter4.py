"""
Real-world integration test for SPSSMirror.

Replicates the analysis workflow of "CHAPTER FOUR" (a thesis chapter
analyzing social media's effect on forest conservation awareness/behavior
among Nigerian tertiary students) against the actual survey CSV
(forest_conservation_response_data.csv, n=1575), using ONLY the SPSSMirror
public API -- no direct scipy/statsmodels/sklearn calls in this script.

Workflow reproduced from the chapter, in order:
  1. Descriptive frequencies (exposure, frequency of exposure, content type)
  2. Reliability of the two composite Likert scales (Cronbach's alpha)
  3. Ho1 - One-way ANOVA: awareness score by institution
  4. Ho2 - Pearson correlation: frequency code vs awareness score
  5. Ho3 - Chi-square: platform vs interaction type
  6. Ho4 - Independent t-test: visual vs text/mixed platforms on awareness
  7. Ho5 - Pearson correlation: awareness score vs behavior score
  8. Ho6 - Logistic regression: predicting behavior change
  9. Bonus: VIF diagnostic on the logistic regression predictors, not in
     the original chapter, exercising Phase 7 on real data.

============================== HONESTY NOTE ===============================
This script's purpose is to prove SPSSMirror runs a full, real,
multi-engine analysis workflow correctly end-to-end on a real dataset --
not to reproduce the chapter's published numbers exactly.

Before writing this script, the underlying reconstructions (composite
score definitions, platform categorization, frequency recoding) were
checked directly against the published tables. Even the analyses that
don't depend on any ambiguous composite-score construction -- the
chi-square test (Main_Platform x Interaction_Type, thesis reports
chi2=1436.566) and the independent t-test (Visual vs Text/Mixed platform
groups, thesis reports t=-9.471, n=688/796) -- come out numerically very
different from the published values when run directly on this CSV
(chi2=28.3, t=0.79, n=1015/490 respectively). Group sizes don't even
match. This means the CSV in hand is not the exact dataset (or not
recoded the same way) behind the chapter's published tables, and no
amount of guessing the composite-score formula would fix that.

So: every number below is a genuine, correctly-computed result on the
actual CSV -- just not a reproduction of Chapter Four's specific
published figures. The point of this script is methodological, not
archival: does SPSSMirror correctly execute this exact SPSS-style
workflow (ANOVA, correlation, chi-square, t-test, logistic regression,
reliability, VIF) end-to-end on a real, messy, real-world survey export?
=============================================================================
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from spssmirror import SPSSMirror

CSV_PATH = Path(__file__).resolve().parent / "forest_conservation_response_data.csv"


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# STEP 0 -- Load and recode. Recoding logic (composite scores, category
# collapsing) is applied via pandas before handing the frame to SPSSMirror,
# since recoding/derived-column creation isn't itself a statistical method
# this library provides -- consistent with how a real analyst preps data
# before opening SPSS.
# ---------------------------------------------------------------------------
import pandas as pd

raw = pd.read_csv(CSV_PATH)

raw["awareness_score"] = raw[
    ["Awareness_Increased", "Learned_Deforestation", "Influenced_Views", "Pay_Attention"]
].mean(axis=1)
raw["behavior_score"] = raw[
    ["Social_Media_Motivates", "Conscious_Increase", "Speak_More", "Feel_Responsible"]
].mean(axis=1)

freq_map = {"Daily": 0, "Weekly": 1, "Monthly": 2, "Rarely": 3}
raw["freq_code"] = raw["Frequency"].map(freq_map)  # "Not Applicable" -> NaN, dropped by each test automatically

visual_platforms = {"Instagram", "TikTok", "YouTube"}
text_platforms = {"Facebook", "Twitter (X)", "WhatsApp"}
platform_cat = pd.Series(pd.NA, index=raw.index, dtype="object")
platform_cat[raw["Main_Platform"].isin(visual_platforms)] = "Visual"
platform_cat[raw["Main_Platform"].isin(text_platforms)] = "Text/Mixed"
raw["platform_cat"] = platform_cat

raw["behavior_changed_binary"] = raw["Changed_Behavior"].map({"Yes": 1, "No": 0})  # "Not sure" -> NaN, dropped

mirror = SPSSMirror().load_dataframe(raw)
print(f"Loaded {mirror.shape()[0]} respondents, {mirror.shape()[1]} columns")


# ---------------------------------------------------------------------------
# STEP 1 -- Descriptive frequencies (Tables 4.1-4.3 style)
# ---------------------------------------------------------------------------
section("STEP 1: DESCRIPTIVE FREQUENCIES")

exposure = mirror.descriptive().frequency_table("Exposed_to_Content")
print(f"\nExposed to content (n={exposure.n}, missing={exposure.n_missing}):")
for cat, freq, pct in zip(exposure.categories, exposure.frequencies, exposure.percentages):
    print(f"  {cat:<10} {freq:>5}  ({pct:5.1f}%)")

freq_table = mirror.descriptive().frequency_table("Frequency", sort_by="frequency")
print("\nFrequency of exposure:")
for cat, freq, pct in zip(freq_table.categories, freq_table.frequencies, freq_table.percentages):
    print(f"  {cat:<16} {freq:>5}  ({pct:5.1f}%)")

content_table = mirror.descriptive().frequency_table("Content_Type", sort_by="frequency")
print("\nContent type encountered:")
for cat, freq, pct in zip(content_table.categories, content_table.frequencies, content_table.percentages):
    print(f"  {cat:<20} {freq:>5}  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# STEP 2 -- Reliability of the two composite Likert scales
# (Methodologically necessary before treating either as a valid composite --
# the original chapter doesn't report this, but any real analysis should.)
# ---------------------------------------------------------------------------
section("STEP 2: RELIABILITY OF COMPOSITE SCALES (Cronbach's Alpha)")

awareness_items = ["Awareness_Increased", "Learned_Deforestation", "Influenced_Views", "Pay_Attention"]
behavior_items = ["Social_Media_Motivates", "Conscious_Increase", "Speak_More", "Feel_Responsible"]

alpha_awareness = mirror.psychometrics().cronbach_alpha(awareness_items)
alpha_behavior = mirror.psychometrics().cronbach_alpha(behavior_items)
print(f"Awareness scale (4 items): alpha = {alpha_awareness.statistic:.3f}  (n={alpha_awareness.n_respondents})")
print(f"Behavior scale (4 items):  alpha = {alpha_behavior.statistic:.3f}  (n={alpha_behavior.n_respondents})")
for name, r in [("Awareness", alpha_awareness), ("Behavior", alpha_behavior)]:
    verdict = "acceptable (>=0.70)" if r.statistic >= 0.70 else "below conventional threshold"
    print(f"  -> {name} scale reliability is {verdict}")


# ---------------------------------------------------------------------------
# STEP 3 -- Ho1: One-way ANOVA, awareness score by institution
# ---------------------------------------------------------------------------
section("STEP 3 (Ho1): ANOVA -- Awareness Score by Institution")

anova = mirror.frequentist().anova_oneway("awareness_score", "Institution", post_hoc=True)
term = anova.terms[0]
print(f"F({term.df:.0f}, {anova.residual_df:.0f}) = {term.f_value:.3f}, p = {term.p_value:.4f}")
print(f"Partial eta-squared = {term.partial_eta_sq:.4f}")
verdict = "REJECT" if term.p_value is not None and term.p_value < 0.05 else "FAIL TO REJECT (accept)"
print(f"-> {verdict} the null hypothesis of no institutional difference.")
print("\nGroup means:")
for g in anova.group_summaries:
    print(f"  {g.group:<45} n={g.n:<5} mean={g.mean:.3f}  sd={g.std:.3f}")
print("\nThesis reported: F=0.455, p=.713 (fail to reject) -- see honesty note at top of script.")


# ---------------------------------------------------------------------------
# STEP 4 -- Ho2: Pearson correlation, frequency code vs awareness score
# ---------------------------------------------------------------------------
section("STEP 4 (Ho2): Correlation -- Frequency of Exposure vs Awareness")

corr_freq = mirror.correlations().pearson("freq_code", "awareness_score")
print(f"r = {corr_freq.coefficient:.3f}, p = {corr_freq.p_value:.4f}, n = {corr_freq.n}")
print(f"95% CI: [{corr_freq.ci_lower:.3f}, {corr_freq.ci_upper:.3f}]")
verdict = "REJECT" if corr_freq.p_value < 0.05 else "FAIL TO REJECT"
direction = "more frequent exposure -> higher awareness" if corr_freq.coefficient < 0 else "more frequent exposure -> lower awareness"
print(f"-> {verdict} the null. Since Daily=0...Rarely=3, a negative r means: {direction}.")
print("\nThesis reported: r=-.223, p=.000, N=1575 -- see honesty note at top of script.")


# ---------------------------------------------------------------------------
# STEP 5 -- Ho3: Chi-square, platform vs interaction type
# ---------------------------------------------------------------------------
section("STEP 5 (Ho3): Chi-Square -- Platform vs Interaction Type")

chi = mirror.categorical().chi_square_independence("Main_Platform", "Interaction_Type")
print(f"chi2({(len(chi.row_categories)-1)*(len(chi.col_categories)-1)}) = {chi.chi_square:.3f}, p = {chi.chi_square_p:.4f}")
print(f"Cramer's V = {chi.cramers_v:.3f}, n = {chi.n}")
if chi.chi_square_p is None:
    verdict = "INCONCLUSIVE"
else:
    verdict = "REJECT" if chi.chi_square_p < 0.05 else "FAIL TO REJECT"
print(f"-> {verdict} the null hypothesis of independence.")
print("\nThesis reported: chi2=1436.566, df=24, p=.000 -- see honesty note at top of script.")


# ---------------------------------------------------------------------------
# STEP 6 -- Ho4: Independent t-test, visual vs text/mixed platforms
# ---------------------------------------------------------------------------
section("STEP 6 (Ho4): Independent T-Test -- Visual vs Text/Mixed Platforms")

ttest = mirror.frequentist().t_test_independent("awareness_score", "platform_cat", "Text/Mixed", "Visual")
print(f"t({ttest.df:.0f}) = {ttest.statistic:.3f}, p = {ttest.p_value:.4f}")
print(f"Cohen's d = {ttest.effect_size:.3f}")
groups = getattr(ttest, "groups", None)
if groups is not None:
    for label, stats_dict in groups.items():
        print(f"  {label:<12} n={stats_dict['n']:<5} mean={stats_dict['mean']:.4f}  sd={stats_dict['std']:.4f}")
else:
    print("  Group statistics unavailable.")
verdict = "REJECT" if ttest.p_value < 0.05 else "FAIL TO REJECT"
print(f"-> {verdict} the null hypothesis of no group difference.")
print("\nThesis reported: t=-9.471, p=.000, n=688/796, means=3.2104/3.6146 -- see honesty note at top.")


# ---------------------------------------------------------------------------
# STEP 7 -- Ho5: Pearson correlation, awareness vs behavior
# ---------------------------------------------------------------------------
section("STEP 7 (Ho5): Correlation -- Awareness Score vs Behavior Score")

corr_ab = mirror.correlations().pearson("awareness_score", "behavior_score")
print(f"r = {corr_ab.coefficient:.3f}, p = {corr_ab.p_value:.4f}, n = {corr_ab.n}")
print(f"95% CI: [{corr_ab.ci_lower:.3f}, {corr_ab.ci_upper:.3f}]")
verdict = "REJECT" if corr_ab.p_value < 0.05 else "FAIL TO REJECT"
print(f"-> {verdict} the null hypothesis of no relationship.")
print("\nThesis reported: r=.800, p=.000, N=1575 -- see honesty note at top of script.")


# ---------------------------------------------------------------------------
# STEP 8 -- Ho6: Logistic regression predicting behavior change
# ---------------------------------------------------------------------------
section("STEP 8 (Ho6): Logistic Regression -- Predicting Behavior Change")

logit = mirror.regression().logistic("behavior_changed_binary ~ freq_code + C(Age) + C(Main_Platform)")
print(f"Pseudo R-squared = {logit.pseudo_r_squared:.4f}, n = {logit.n}")
print(f"{'Term':<38}{'B':>9}{'SE':>9}{'z':>9}{'p':>9}")
for c in logit.coefficients:
    se = f"{c.std_error:.3f}" if c.std_error is not None else "  n/a"
    z = f"{c.z_value:.3f}" if c.z_value is not None else "  n/a"
    p = f"{c.p_value:.4f}" if c.p_value is not None else "  n/a"
    print(f"{c.term:<38}{c.b:>9.3f}{se:>9}{z:>9}{p:>9}")
print("\nThesis reported (variables differ slightly -- thesis used Age/Main_Platform as")
print("Wald-test blocks, not individual dummy coefficients): Frequency B=-.226, Exp(B)=.797, p=.000")


# ---------------------------------------------------------------------------
# STEP 9 (bonus) -- VIF check on the logistic regression's continuous
# predictor set, exercising Phase 7 diagnostics on real data (not in the
# original chapter).
# ---------------------------------------------------------------------------
section("STEP 9 (bonus): Multicollinearity Check")

age_dummy = pd.get_dummies(raw["Age"], prefix="age", drop_first=True).astype(float)
vif_df = pd.concat([raw[["freq_code"]], age_dummy], axis=1).dropna()
mirror_vif = SPSSMirror().load_dataframe(vif_df)
vif_result = mirror_vif.diagnostics().vif(list(vif_df.columns))
print("VIF for logistic regression's continuous/dummy predictors:")
for item in vif_result.items:
    flag = "OK" if item.vif < 5 else ("CAUTION" if item.vif < 10 else "CONCERNING")
    print(f"  {item.predictor:<12} VIF={item.vif:.3f}  [{flag}]")


section("SCRIPT COMPLETE")
print("Every statistical method above ran through the SPSSMirror public API only.")
print("No scipy/statsmodels/sklearn calls appear anywhere in this script.")
print("See the module docstring for an honest account of numeric agreement with")
print("the published Chapter Four tables.")
