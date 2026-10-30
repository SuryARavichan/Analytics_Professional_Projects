"""
causal_inference.py — a difference-in-differences (DiD) estimator for
Flowdesk's staged "Smart Templates" feature rollout, which shipped
region-by-region rather than via randomized assignment (a very common
real-world constraint: infra/ops rollouts, regulatory approval timing, and
sales-territory-driven launches routinely rule out a clean A/B test).

A two-way fixed-effects (TWFE) specification is used:

    rate_rt = alpha_r + gamma_t + beta * (treated_r * post_t) + e_rt

where alpha_r are region fixed effects (absorbing any constant regional
gap), gamma_t are week fixed effects (absorbing any common time trend
affecting every region, e.g. general product improvement), and beta -- the
coefficient on the treated-region x post-period interaction -- is the DiD
causal estimate. This identifies the treatment effect ONLY under the
parallel-trends assumption (treated and control regions would have moved
in parallel absent treatment), which is why parallel_trends_check() and
placebo_test() exist: a DiD estimate without checking that assumption is
not a trustworthy causal estimate, and this module treats the check as a
first-class deliverable, not an afterthought.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class DidResult:
    effect_pp: float
    se: float
    p_value: float
    ci_95: tuple
    model_summary: str


def diff_in_diff(panel: pd.DataFrame, outcome_col: str = "weekly_active_rate") -> DidResult:
    df = panel.copy()
    df["treated_post"] = (df.treated_region & df.post).astype(int)
    model = smf.ols(f"{outcome_col} ~ C(region) + C(week) + treated_post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["region"]}
    )
    coef = model.params["treated_post"]
    se = model.bse["treated_post"]
    p = model.pvalues["treated_post"]
    ci = model.conf_int().loc["treated_post"].tolist()
    return DidResult(effect_pp=coef * 100, se=se * 100, p_value=p, ci_95=(ci[0] * 100, ci[1] * 100),
                      model_summary=model.summary().as_text())


def parallel_trends_check(panel: pd.DataFrame, rollout_week: int, outcome_col: str = "weekly_active_rate") -> pd.DataFrame:
    """Pre-period-only trend comparison: fits a separate linear week-trend
    for treated vs. control regions using ONLY weeks before the rollout,
    and returns the slope difference -- a large, significant slope gap in
    the pre-period would indicate the parallel-trends assumption is
    violated and the DiD estimate above should not be trusted."""
    pre = panel[panel.week < rollout_week].copy()
    rows = []
    for is_treated, label in [(True, "treated"), (False, "control")]:
        sub = pre[pre.treated_region == is_treated]
        slope, intercept = np.polyfit(sub.week, sub[outcome_col], 1)
        rows.append({"group": label, "pre_period_weekly_slope": slope, "intercept": intercept})
    return pd.DataFrame(rows)


def placebo_test(panel: pd.DataFrame, fake_rollout_week: int, real_rollout_week: int,
                  outcome_col: str = "weekly_active_rate") -> DidResult:
    """Re-runs the DiD spec using ONLY pre-period data (weeks before the
    real rollout), with a fabricated rollout week inserted in the middle
    of that pre-period. If the DiD design is valid, this fake treatment
    effect should NOT be statistically significant -- a significant
    placebo effect would be a red flag that something other than the real
    treatment is driving the main result."""
    pre_only = panel[panel.week < real_rollout_week].copy()
    pre_only["post"] = pre_only.week >= fake_rollout_week
    return diff_in_diff(pre_only, outcome_col=outcome_col)
