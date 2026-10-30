"""
hte.py — heterogeneous treatment effect (HTE) / subgroup analysis: does
the treatment effect actually vary by segment, or is the segment-level
variation just noise?

Method: reconstruct individual-level rows (1 row per user, a `converted`
0/1 column) from each segment's aggregated (n, conversions) counts per
arm -- exactly the granularity a real event-level dataset would have --
then fit a single logistic regression `converted ~ treatment * C(segment)`
across ALL segments at once. The treatment:segment interaction term
p-values are the right way to ask "does the effect differ by segment,"
rather than eyeballing whether each segment's separate point estimate
looks different (which inflates false-discovery risk through multiple
implicit comparisons and doesn't formally test the difference itself).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _expand_segment_to_rows(segment: dict) -> pd.DataFrame:
    rows = []
    for arm, n_key, conv_key in [("control", "control_n", "control_conversions"),
                                  ("treatment", "treatment_n", "treatment_conversions")]:
        n, conv = segment[n_key], segment[conv_key]
        rows.append(pd.DataFrame({
            "segment": segment["segment"], "treatment": 1 if arm == "treatment" else 0,
            "converted": [1] * conv + [0] * (n - conv),
        }))
    return pd.concat(rows, ignore_index=True)


@dataclass
class HteResult:
    segment_effects: pd.DataFrame
    interaction_p_values: dict
    any_significant_interaction: bool
    model_summary: str


def test_heterogeneous_effects(segments: list[dict], reference_segment: str, alpha: float = 0.05) -> HteResult:
    df = pd.concat([_expand_segment_to_rows(s) for s in segments], ignore_index=True)
    df["segment"] = pd.Categorical(df["segment"], categories=[reference_segment] + [
        s["segment"] for s in segments if s["segment"] != reference_segment])

    model = smf.logit("converted ~ treatment * C(segment)", data=df).fit(disp=0)

    interaction_terms = {k: v for k, v in model.pvalues.items() if ":" in k}
    any_sig = any(p < alpha for p in interaction_terms.values())

    rows = []
    for s in segments:
        p_c = s["control_conversions"] / s["control_n"]
        p_t = s["treatment_conversions"] / s["treatment_n"]
        rows.append({"segment": s["segment"], "control_rate": round(p_c, 4), "treatment_rate": round(p_t, 4),
                      "lift_pp": round((p_t - p_c) * 100, 2)})

    return HteResult(
        segment_effects=pd.DataFrame(rows), interaction_p_values=interaction_terms,
        any_significant_interaction=any_sig, model_summary=model.summary().as_text(),
    )
