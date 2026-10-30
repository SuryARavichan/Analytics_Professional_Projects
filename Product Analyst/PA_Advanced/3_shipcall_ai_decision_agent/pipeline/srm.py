"""
srm.py — Sample Ratio Mismatch (SRM) check: a chi-square goodness-of-fit
test comparing the ACTUAL control/treatment split to the INTENDED split.

This is one of the most respected, most-often-skipped checks in applied
experimentation (see Fabijan et al. 2019, and Ron Kohavi's writing on
trustworthy experimentation): if the randomization/bucketing pipeline has
a bug, the two arms can end up systematically different from each other
in ways that have nothing to do with the treatment -- which invalidates
EVERY other result from the experiment, including a seemingly-significant
primary-metric win. Per standard practice, this check uses a stricter
significance threshold (alpha=0.001, not 0.05), because even a healthy
randomization system produces a marginal p-value (e.g., 0.02-0.04)
occasionally by chance, and treating that as SRM would create constant
false alarms.
"""
from dataclasses import dataclass
from scipy import stats


@dataclass
class SrmResult:
    observed_ratio: float
    expected_ratio: float
    chi2: float
    p_value: float
    is_srm: bool


def check_srm(control_n: int, treatment_n: int, expected_ratio: float = 0.5, alpha: float = 0.001) -> SrmResult:
    total = control_n + treatment_n
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    chi2, p = stats.chisquare([control_n, treatment_n], f_exp=[expected_control, expected_treatment])
    return SrmResult(
        observed_ratio=control_n / total, expected_ratio=expected_ratio,
        chi2=chi2, p_value=p, is_srm=p < alpha,
    )
