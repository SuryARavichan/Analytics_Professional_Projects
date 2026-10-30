"""
voice_metrics.py — containment (call resolved without a human dispatcher)
and transcription-quality summaries by region/version, plus the
two-proportion z-test used to compare a voice-model release's effect on
one region against the nationwide aggregate -- the same "does the
aggregate hide a segment" check used elsewhere in this session, applied
here to a genuinely different and well-documented voice-AI risk:
regional/dialectal ASR coverage gaps.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


def containment_summary(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    return df.groupby(groupby).agg(
        n=("call_id", "count"),
        containment=("escalated_to_human", lambda s: 1 - s.mean()),
        avg_confidence=("transcription_confidence", "mean"),
        avg_interruptions=("n_interruptions", "mean"),
        avg_latency_ms=("avg_response_latency_ms", "mean"),
        avg_sentiment=("driver_sentiment_score", "mean"),
    ).round(4)


@dataclass
class ContainmentTestResult:
    p_before: float
    p_after: float
    lift_pp: float
    z: float
    p_value: float
    n_before: int
    n_after: int
    significant: bool


def containment_shift_test(df: pd.DataFrame, region: str | None = None, alpha: float = 0.01) -> ContainmentTestResult:
    """Two-proportion z-test on containment rate, v1 vs v2. region=None
    tests the NATIONWIDE aggregate; passing a region tests that region
    alone -- running both is what surfaces the gap between them."""
    sub = df if region is None else df[df.driver_region == region]
    v1, v2 = sub[sub.model_version == "v1"], sub[sub.model_version == "v2"]
    s1, s2 = int((~v1.escalated_to_human).sum()), int((~v2.escalated_to_human).sum())
    n1, n2 = len(v1), len(v2)
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return ContainmentTestResult(
        p_before=round(p1, 4), p_after=round(p2, 4), lift_pp=round((p2 - p1) * 100, 2),
        z=round(z, 3), p_value=p_value, n_before=n1, n_after=n2, significant=bool(p_value < alpha),
    )
