"""
agent.py — perceive -> act -> reason -> report, applied to Meridian
Health AI's embedding-model release review (v1 -> v2), run separately
for each query segment. The rubric is explicit and safety-first: a
segment is only reported as improved if retrieval quality improved
significantly; a segment where retrieval quality REGRESSED significantly
must be flagged as a patient-safety risk regardless of how the
overall/aggregate category is trending, since the aggregate is exactly
what hides this in this dataset's headline story.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from llm_client import LLMClient  # noqa: E402
from retrieval_metrics import compare_versions_ttest, two_proportion_ztest  # noqa: E402
from hallucination_risk import silent_failure_rate  # noqa: E402

SYSTEM_PROMPT = (
    "You are GroundTruth Clinical, a RAG-quality analytics agent reviewing an embedding-model release "
    "(v1 -> v2) for Meridian Health AI, a clinical-documentation copilot. For each query segment you are "
    "given a t-test on precision@k (retrieval quality) v1 vs v2, a two-proportion test on hallucination "
    "rate, and the silent-failure rate (hallucination occurred AND the physician accepted the answer as-is) "
    "for both versions. Apply this rubric:\n"
    "- If precision@k improved significantly (p<0.01) and did not worsen hallucination rate: IMPROVED.\n"
    "- If precision@k REGRESSED significantly (p<0.01): PATIENT SAFETY RISK, regardless of how any broader "
    "aggregate category is trending -- a segment-level regression is never excused by an aggregate that "
    "looks fine, and you must say so explicitly.\n"
    "- Otherwise: NO SIGNIFICANT CHANGE.\n"
    "Write a short release-review note for a Clinical Informatics / Patient Safety lead with: (1) VERDICT, "
    "(2) RETRIEVAL QUALITY EVIDENCE, (3) DOWNSTREAM SAFETY EVIDENCE (hallucination rate, silent failure "
    "rate), (4) RECOMMENDED ACTION. Only draw conclusions the evidence actually supports."
)


@dataclass
class SegmentReviewReport:
    segment_label: str
    verdict: str
    precision_result: dict
    hallucination_result: dict
    silent_failure_v1: float
    silent_failure_v2: float
    narrative: str = ""
    llm_meta: dict = field(default_factory=dict)


def review_segment(queries: pd.DataFrame, segment_label: str, query_type: str, rare_flag: bool | None,
                    client: LLMClient) -> SegmentReviewReport:
    precision_result = compare_versions_ttest(queries, "precision_at_k", filter_query_type=query_type, filter_rare=rare_flag)

    sub = queries[queries.query_type == query_type]
    if rare_flag is not None:
        sub = sub[sub.rare_drug_combination == rare_flag]
    v1, v2 = sub[sub.embedding_version == "v1"], sub[sub.embedding_version == "v2"]
    hallucination_result = two_proportion_ztest(
        int(v1.hallucination_occurred.sum()), len(v1), int(v2.hallucination_occurred.sum()), len(v2),
    )

    sf = silent_failure_rate(sub, groupby=["embedding_version"])
    sf_v1 = float(sf.loc["v1", "silent_failure_rate"]) if "v1" in sf.index else float("nan")
    sf_v2 = float(sf.loc["v2", "silent_failure_rate"]) if "v2" in sf.index else float("nan")

    if precision_result["significant"] and precision_result["diff"] < 0:
        verdict = "PATIENT_SAFETY_RISK"
    elif precision_result["significant"] and precision_result["diff"] > 0:
        verdict = "IMPROVED"
    else:
        verdict = "NO_SIGNIFICANT_CHANGE"

    evidence = (
        f"Segment: {segment_label}\n"
        f"Rubric verdict: {verdict}\n\n"
        f"Precision@k t-test (v1 vs v2): {precision_result}\n\n"
        f"Hallucination-rate two-proportion test (v1 vs v2): {hallucination_result}\n\n"
        f"Silent failure rate: v1={sf_v1:.4f}, v2={sf_v2:.4f}\n"
    )
    resp = client.complete(prompt=evidence, system=SYSTEM_PROMPT, prompt_id=f"segment_review_{segment_label}")

    return SegmentReviewReport(
        segment_label=segment_label, verdict=verdict, precision_result=precision_result,
        hallucination_result=hallucination_result, silent_failure_v1=sf_v1, silent_failure_v2=sf_v2,
        narrative=resp.text, llm_meta={"backend": resp.backend, "cached": resp.cached},
    )
