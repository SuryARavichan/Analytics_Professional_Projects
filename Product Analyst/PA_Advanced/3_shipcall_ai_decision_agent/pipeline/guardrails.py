"""
guardrails.py — checks each experiment's guardrail metrics (metrics you
are NOT trying to move, but must not let get worse -- e.g. churn, support
load) using the same two-proportion z-test as the primary metric, flagging
a guardrail as VIOLATED only when it moved significantly in the bad
direction (per its declared "lower_is_better" / "higher_is_better" sense).
A guardrail that's flat, or moved favorably, is not a violation even if
technically "significant" -- only a significant move in the harmful
direction counts.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class GuardrailResult:
    name: str
    control_rate: float
    treatment_rate: float
    direction: str
    delta_pp: float
    p_value: float
    violated: bool


def check_guardrail(name: str, control_n: int, control_events: int, treatment_n: int, treatment_events: int,
                     direction: str = "lower_is_better", alpha: float = 0.05) -> GuardrailResult:
    p_c = control_events / control_n
    p_t = treatment_events / treatment_n
    p_pool = (control_events + treatment_events) / (control_n + treatment_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p_t - p_c) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    moved_worse = (p_t > p_c) if direction == "lower_is_better" else (p_t < p_c)
    violated = (p_value < alpha) and moved_worse

    return GuardrailResult(name=name, control_rate=p_c, treatment_rate=p_t, direction=direction,
                            delta_pp=(p_t - p_c) * 100, p_value=p_value, violated=violated)


def check_all_guardrails(guardrails: list[dict], alpha: float = 0.05) -> list[GuardrailResult]:
    return [check_guardrail(g["name"], g["control_n"], g["control_events"], g["treatment_n"], g["treatment_events"],
                             g.get("direction", "lower_is_better"), alpha) for g in guardrails]
