"""
ab_testing_lite.py — a minimal two-proportion z-test returning a plain
dict (rather than CausalLift's dataclass), used by ShipCall AI's agent to
evaluate each experiment's primary metric before applying the decision
rubric. Kept as its own small module (instead of importing across project
folders) so this project stays independently runnable, matching the rest
of this portfolio's per-project self-containment.
"""
import numpy as np
from scipy import stats


def two_proportion_summary(control_n: int, control_conversions: int, treatment_n: int, treatment_conversions: int) -> dict:
    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p_t - p_c) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "p_control": p_c, "p_treatment": p_t, "lift_pp": (p_t - p_c) * 100,
        "z": z, "p_value": p_value, "significant": p_value < 0.05,
    }
