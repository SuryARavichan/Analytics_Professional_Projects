"""
agent.py — orchestrates the "perceive -> act -> reason -> report" loop:

  PERCEIVE  a flagged anomaly (from anomaly_detector.py)
  ACT       run a fixed diagnostic playbook (diagnostics.py: breakdown by
            region, breakdown by channel, compare to a baseline period) --
            deterministic, real SQL, always runs the same way for every
            incident so results are reproducible and auditable
  REASON    an LLM call synthesizes the diagnostic evidence (not the raw
            question) into a root-cause hypothesis, confidence level, and
            recommended action
  REPORT    a structured IncidentReport the dashboard/notebook renders

SCOPING NOTE (worth saying out loud in an interview): this is a fixed
diagnostic playbook triggered on detection, not a dynamic tool-selection
ReAct loop where the model chooses which query to run next. That's a
deliberate scope decision for a reliable, auditable v1 -- every incident
gets the same evidence-gathering steps, so the system's behavior doesn't
depend on the model's tool-choice quality. A natural v2 extension is
letting the model choose which diagnostic to run next (and how many),
useful when the fixed playbook doesn't localize the cause on the first
pass.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from llm_client import LLMClient  # noqa: E402
from diagnostics import breakdown_by_region, breakdown_by_channel, compare_to_baseline  # noqa: E402

SYSTEM_PROMPT = (
    "You are SentinelBI, an autonomous analytics agent that investigates flagged revenue "
    "anomalies for an e-commerce business. You are given: the anomaly detection stats, and "
    "the results of 3 diagnostic queries (region breakdown, channel breakdown, comparison to "
    "a baseline period). Write a short incident report with: (1) ROOT CAUSE hypothesis, "
    "(2) CONFIDENCE (High/Medium/Low) with one-sentence justification, (3) RECOMMENDED ACTION. "
    "Only draw conclusions the evidence actually supports. If evidence points to a broad, "
    "uniform movement across all regions/channels rather than one isolated slice, say so and "
    "downgrade confidence in any single-cause story."
)


@dataclass
class IncidentReport:
    incident_id: str
    kind: str  # "point_anomaly" | "regional_drift"
    window: tuple
    detection_stats: dict
    region_breakdown: "pd.DataFrame"
    channel_breakdown: "pd.DataFrame" = None
    baseline_comparison: "pd.DataFrame" = None
    narrative: str = ""
    llm_meta: dict = field(default_factory=dict)


def investigate(db_path: str, incident_id: str, kind: str, start_date: str, end_date: str,
                 baseline_start: str, baseline_end: str, detection_stats: dict,
                 client: LLMClient) -> IncidentReport:
    # ACT: run the fixed diagnostic playbook (real SQL, always the same steps)
    region_bd = breakdown_by_region(db_path, start_date, end_date)
    channel_bd = breakdown_by_channel(db_path, start_date, end_date) if kind == "point_anomaly" else None
    baseline_cmp = compare_to_baseline(db_path, start_date, end_date, baseline_start, baseline_end,
                                        group_col="region")

    # REASON: LLM synthesizes the gathered evidence
    evidence = (
        f"Detection stats: {detection_stats}\n\n"
        f"Region breakdown for the incident window:\n{region_bd.to_string(index=False)}\n\n"
        + (f"Channel breakdown for the incident window:\n{channel_bd.to_string(index=False)}\n\n" if channel_bd is not None else "")
        + f"Region-level comparison vs. baseline period ({baseline_start} to {baseline_end}):\n{baseline_cmp.to_string(index=False)}\n"
    )
    resp = client.complete(prompt=evidence, system=SYSTEM_PROMPT, prompt_id=incident_id)

    return IncidentReport(
        incident_id=incident_id, kind=kind, window=(start_date, end_date),
        detection_stats=detection_stats, region_breakdown=region_bd,
        channel_breakdown=channel_bd, baseline_comparison=baseline_cmp,
        narrative=resp.text, llm_meta={"backend": resp.backend, "cached": resp.cached},
    )
