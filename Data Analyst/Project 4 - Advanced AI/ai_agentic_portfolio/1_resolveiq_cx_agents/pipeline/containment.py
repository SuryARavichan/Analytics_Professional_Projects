"""
containment.py — the core statistical engine for ResolveIQ's central
finding: Ravel AI's own agent reports a ticket as "contained" (resolved
without a human) the moment it closes the conversation, but a resolution
that doesn't actually fix the customer's problem shows up later as a
REOPENED ticket. This module computes both the agent's OFFICIAL
containment rate and a TRUE, reopen-adjusted containment rate, and tests
whether a change in one implies a change in the other -- because a
policy change can move the official metric without moving the true one,
which is exactly the failure mode an AgentOps analyst has to catch
before reporting a policy change as a win.

The two-proportion z-test here is the same primitive used across this
session's other experimentation work (CausalLift, SentryOps) --
duplicated rather than cross-imported so this project stays independently
runnable, the established self-containment convention in this portfolio.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


def official_containment(df: pd.DataFrame) -> float:
    return (df.resolution_path == "agent_resolved").mean()


def true_containment(df: pd.DataFrame) -> float:
    """Contained AND not reopened within 48h -- the outcome that actually
    matters to the customer, not just what the agent claims."""
    contained = df.resolution_path == "agent_resolved"
    not_reopened = ~df.reopened_within_48h.fillna(False)
    return (contained & not_reopened).mean()


def reopen_rate(df: pd.DataFrame) -> float:
    contained = df[df.resolution_path == "agent_resolved"]
    return contained.reopened_within_48h.mean() if len(contained) else float("nan")


def containment_summary(df: pd.DataFrame, groupby: str = "category") -> pd.DataFrame:
    return df.groupby(groupby).apply(
        lambda g: pd.Series({
            "n_tickets": len(g),
            "official_containment": official_containment(g),
            "reopen_rate": reopen_rate(g),
            "true_containment": true_containment(g),
        }), include_groups=False
    ).round(4)


@dataclass
class ProportionTestResult:
    p_before: float
    p_after: float
    lift_pp: float
    z: float
    p_value: float
    n_before: int
    n_after: int
    significant: bool


def two_proportion_ztest(before_successes: int, before_n: int, after_successes: int, after_n: int,
                          alpha: float = 0.01) -> ProportionTestResult:
    p_b, p_a = before_successes / before_n, after_successes / after_n
    p_pool = (before_successes + after_successes) / (before_n + after_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / before_n + 1 / after_n))
    z = (p_a - p_b) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return ProportionTestResult(
        p_before=round(p_b, 4), p_after=round(p_a, 4), lift_pp=round((p_a - p_b) * 100, 2),
        z=round(z, 3), p_value=p_value, n_before=before_n, n_after=after_n, significant=bool(p_value < alpha),
    )


def containment_shift_test(df: pd.DataFrame, category: str, metric: str = "official") -> ProportionTestResult:
    """metric: 'official' tests official_containment v1 vs v2;
    'true' tests true_containment v1 vs v2 for the same category --
    running both side by side is the whole point of this module."""
    sub = df[df.category == category]
    v1, v2 = sub[sub.agent_version == "v1"], sub[sub.agent_version == "v2"]
    if metric == "official":
        s1, s2 = (v1.resolution_path == "agent_resolved").sum(), (v2.resolution_path == "agent_resolved").sum()
    elif metric == "true":
        s1 = ((v1.resolution_path == "agent_resolved") & (~v1.reopened_within_48h.fillna(False))).sum()
        s2 = ((v2.resolution_path == "agent_resolved") & (~v2.reopened_within_48h.fillna(False))).sum()
    else:
        raise ValueError("metric must be 'official' or 'true'")
    return two_proportion_ztest(s1, len(v1), s2, len(v2))


def classify_policy_change(official_result: ProportionTestResult, true_result: ProportionTestResult) -> str:
    """The rubric that turns two test results into one verdict:
      - official improved significantly AND true improved significantly -> GENUINE_IMPROVEMENT
      - official improved significantly but true did NOT -> FALSE_IMPROVEMENT (metric gaming)
      - true declined significantly -> REGRESSION
      - neither moved significantly -> NO_CHANGE
    """
    official_up = official_result.significant and official_result.lift_pp > 0
    true_up = true_result.significant and true_result.lift_pp > 0
    true_down = true_result.significant and true_result.lift_pp < 0

    if true_down:
        return "REGRESSION"
    if official_up and true_up:
        return "GENUINE_IMPROVEMENT"
    if official_up and not true_up:
        return "FALSE_IMPROVEMENT"
    return "NO_CHANGE"
