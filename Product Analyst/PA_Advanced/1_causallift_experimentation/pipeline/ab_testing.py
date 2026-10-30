"""
ab_testing.py — core statistical primitives for CausalLift's fixed-horizon
experiment analysis: a two-proportion z-test for binary metrics, and a
CUPED-adjusted (Controlled-experiment Using Pre-Experiment Data) t-test for
continuous metrics.

CUPED (Deng et al., 2013, Microsoft) uses a covariate measured BEFORE the
experiment started (so it cannot be affected by treatment) to strip
predictable, treatment-unrelated variance out of the outcome metric before
testing -- the same logic as ANCOVA / regression adjustment, popularized in
industry AB-testing platforms because it lets you detect the same effect
size with less traffic, or detect a smaller effect with the same traffic.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ProportionTestResult:
    p_control: float
    p_treatment: float
    lift_pp: float
    lift_relative_pct: float
    z: float
    p_value: float
    ci_95: tuple


def two_proportion_ztest(control_successes: int, control_n: int,
                          treatment_successes: int, treatment_n: int) -> ProportionTestResult:
    p_c = control_successes / control_n
    p_t = treatment_successes / treatment_n
    p_pool = (control_successes + treatment_successes) / (control_n + treatment_n)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p_t - p_c) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)
    diff = p_t - p_c
    ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    return ProportionTestResult(
        p_control=p_c, p_treatment=p_t, lift_pp=diff * 100,
        lift_relative_pct=100 * diff / p_c if p_c > 0 else float("nan"),
        z=z, p_value=p_value, ci_95=ci,
    )


@dataclass
class CupedResult:
    theta: float
    raw_variance: float
    adjusted_variance: float
    variance_reduction_pct: float
    raw_t: float
    raw_p: float
    adjusted_t: float
    adjusted_p: float
    raw_lift: float
    adjusted_lift: float


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y: outcome metric, x: pre-experiment covariate (same units/index
    order as y). Returns the CUPED-adjusted outcome: y_cuped = y - theta*(x - mean(x)),
    where theta = Cov(y,x) / Var(x) is fit on the POOLED sample (both arms
    together), which is what makes this an unbiased adjustment -- theta
    does not use treatment-assignment information."""
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def cuped_ttest(y_control: np.ndarray, y_treatment: np.ndarray,
                 x_control: np.ndarray, x_treatment: np.ndarray) -> CupedResult:
    y_all = np.concatenate([y_control, y_treatment])
    x_all = np.concatenate([x_control, x_treatment])
    y_adj_all, theta = cuped_adjust(y_all, x_all)
    y_adj_control = y_adj_all[: len(y_control)]
    y_adj_treatment = y_adj_all[len(y_control):]

    raw_t, raw_p = stats.ttest_ind(y_treatment, y_control, equal_var=False)
    adj_t, adj_p = stats.ttest_ind(y_adj_treatment, y_adj_control, equal_var=False)

    raw_var = np.var(y_all, ddof=1)
    adj_var = np.var(y_adj_all, ddof=1)

    return CupedResult(
        theta=theta, raw_variance=raw_var, adjusted_variance=adj_var,
        variance_reduction_pct=100 * (1 - adj_var / raw_var),
        raw_t=raw_t, raw_p=raw_p, adjusted_t=adj_t, adjusted_p=adj_p,
        raw_lift=y_treatment.mean() - y_control.mean(),
        adjusted_lift=y_adj_treatment.mean() - y_adj_control.mean(),
    )


def required_sample_size(baseline_rate: float, mde_relative: float,
                          alpha: float = 0.05, power: float = 0.80) -> int:
    """Two-proportion required per-arm sample size for a given baseline
    rate and minimum detectable relative lift, via the standard normal-
    approximation formula -- used here to sanity-check that Experiment A's
    12,000-user sample was adequately powered for its observed effect."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pooled = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * pooled * (1 - pooled)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    return int(np.ceil(n))
