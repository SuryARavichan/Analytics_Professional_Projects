"""
agent.py — perceive -> act -> reason -> report for RouteVoice, covering
THREE different incident types with three different detectors:
  - voice_metrics.containment_shift_test(), run per-region against a
    known model deploy, for the regional dialect regression
  - call_anomaly.point_anomaly_detector(), for the shared ASR outage
  - voice_metrics.containment_shift_test() again, run around the benign
    greeting-script deploy, which must be correctly dismissed
The rubric mirrors the one used in the companion SentryOps AI and
ResolveIQ projects: a deploy is only an INCIDENT if the outcome metric
(containment) moved significantly and harmfully -- and a region-level
regression must never be excused by a healthy nationwide aggregate.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from llm_client import LLMClient  # noqa: E402
from voice_metrics import containment_shift_test, containment_summary  # noqa: E402

SYSTEM_PROMPT = (
    "You are RouteVoice, a voice-AI monitoring agent for Portside AI, a logistics voice-agent platform. "
    "You review THREE kinds of signals: (1) a regional containment shift test tied to a voice-model deploy, "
    "(2) a shared-infrastructure point-anomaly signal (an ASR provider issue), (3) a containment shift test "
    "around a planned, non-model deploy (a UI/script change). Apply this rubric:\n"
    "1. For a MODEL DEPLOY signal, per region: if containment REGRESSED significantly (p<0.01), this is an "
    "INCIDENT (fairness/coverage risk) regardless of how the NATIONWIDE aggregate is trending -- never excuse "
    "a regional regression with a healthy national number. If containment improved significantly, report as "
    "a genuine improvement.\n"
    "2. For an ASR/infra point-anomaly signal: classify as INCIDENT (transient, infrastructure) if a flagged "
    "window exists across multiple regions/call types simultaneously.\n"
    "3. For a BENIGN/non-model deploy signal: classify as BENIGN / NO ACTION if containment did NOT change "
    "significantly, even if another metric (call duration, latency) changed a lot -- a planned change to "
    "call mechanics is not a regression unless the outcome metric moved.\n"
    "Write a short report for a Voice AI Ops / Fairness reviewer with: (1) VERDICT, (2) EVIDENCE, "
    "(3) LIKELY MECHANISM (only if evidence supports one), (4) RECOMMENDED ACTION."
)


@dataclass
class VoiceIncidentReport:
    incident_id: str
    kind: str
    verdict: str
    evidence: dict
    narrative: str = ""
    llm_meta: dict = field(default_factory=dict)


def review_regional_deploy(calls: pd.DataFrame, region: str, client: LLMClient) -> VoiceIncidentReport:
    national = containment_shift_test(calls, region=None)
    regional = containment_shift_test(calls, region=region)
    verdict = "INCIDENT_FAIRNESS_RISK" if (regional.significant and regional.lift_pp < 0) else \
              ("GENUINE_IMPROVEMENT" if (regional.significant and regional.lift_pp > 0) else "NO_SIGNIFICANT_CHANGE")

    evidence = {"national": vars(national), "regional": vars(regional)}
    prompt = (
        f"Signal type: model deploy regional review\nRegion: {region}\nRubric verdict: {verdict}\n\n"
        f"Nationwide containment shift (v1 vs v2): {vars(national)}\n\n"
        f"{region} containment shift (v1 vs v2): {vars(regional)}\n"
    )
    resp = client.complete(prompt=prompt, system=SYSTEM_PROMPT, prompt_id=f"regional_deploy_{region}")
    return VoiceIncidentReport(incident_id=f"regional_deploy_{region}", kind="model_deploy", verdict=verdict,
                               evidence=evidence, narrative=resp.text, llm_meta={"backend": resp.backend, "cached": resp.cached})


def review_asr_outage(window: dict, client: LLMClient) -> VoiceIncidentReport:
    prompt = f"Signal type: ASR point anomaly\nRubric verdict: INCIDENT_TRANSIENT\n\nDetection window: {window}\n"
    resp = client.complete(prompt=prompt, system=SYSTEM_PROMPT, prompt_id="asr_outage")
    return VoiceIncidentReport(incident_id="asr_outage", kind="infra_anomaly", verdict="INCIDENT_TRANSIENT",
                               evidence={"window": window}, narrative=resp.text,
                               llm_meta={"backend": resp.backend, "cached": resp.cached})


def review_benign_deploy(calls: pd.DataFrame, deploy_ts: pd.Timestamp, window_days: int, client: LLMClient) -> VoiceIncidentReport:
    before = calls[(calls.timestamp >= deploy_ts - pd.Timedelta(days=window_days)) & (calls.timestamp < deploy_ts)]
    after = calls[(calls.timestamp >= deploy_ts) & (calls.timestamp < deploy_ts + pd.Timedelta(days=window_days))]
    duration_before, duration_after = before.call_duration_sec.mean(), after.call_duration_sec.mean()

    s1, n1 = int((~before.escalated_to_human).sum()), len(before)
    s2, n2 = int((~after.escalated_to_human).sum()), len(after)
    import numpy as np
    from scipy import stats
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    significant = bool(p_value < 0.01)
    verdict = "BENIGN_NO_ACTION" if not significant else ("INCIDENT" if p2 < p1 else "GENUINE_IMPROVEMENT")

    evidence = {"duration_before_sec": round(duration_before, 1), "duration_after_sec": round(duration_after, 1),
                "containment_before": round(p1, 4), "containment_after": round(p2, 4), "z": round(z, 3), "p_value": p_value}
    prompt = f"Signal type: benign/non-model deploy review\nRubric verdict: {verdict}\n\nEvidence: {evidence}\n"
    resp = client.complete(prompt=prompt, system=SYSTEM_PROMPT, prompt_id="benign_script_deploy")
    return VoiceIncidentReport(incident_id="benign_script_deploy", kind="non_model_deploy", verdict=verdict,
                               evidence=evidence, narrative=resp.text, llm_meta={"backend": resp.backend, "cached": resp.cached})
