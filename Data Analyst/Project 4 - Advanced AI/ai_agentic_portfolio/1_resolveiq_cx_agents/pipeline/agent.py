"""
agent.py — perceive -> act -> reason -> report, applied to Ravel AI's
agent-version release review. For EACH category affected by the v1->v2
release, the agent is given both the official-containment shift test AND
the true-containment shift test (containment.py), plus the cost impact
(economics.py), and has to apply an explicit rubric rather than just
reading the headline containment number -- because the headline number is
exactly what's misleading for account_recovery.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from llm_client import LLMClient  # noqa: E402
from containment import containment_shift_test, classify_policy_change  # noqa: E402
from economics import reopen_cost_impact  # noqa: E402

SYSTEM_PROMPT = (
    "You are ResolveIQ, an AgentOps analytics agent reviewing a support-agent release (v1 -> v2) for "
    "Ravel AI, an autonomous AI customer-support-agent platform. For each ticket category you are given: "
    "a two-proportion test on the agent's OFFICIAL self-reported containment rate, a two-proportion test on "
    "TRUE (reopen-within-48h-adjusted) containment, this module's rubric classification, and a cost-impact "
    "breakdown. Apply this rubric strictly:\n"
    "- GENUINE_IMPROVEMENT: report it as a real win.\n"
    "- FALSE_IMPROVEMENT: this means the official metric improved significantly but the TRUE outcome did "
    "NOT improve significantly. You MUST call this out explicitly as a metric-gaming risk, not a real "
    "improvement, regardless of how good the official number looks -- and say plainly that shipping this "
    "as a win to stakeholders would be misleading.\n"
    "- REGRESSION: report it as a real problem needing a fix.\n"
    "- NO_CHANGE: report no significant change either way.\n"
    "Write a short release-review note for a Product/Support Ops leader with: (1) VERDICT (one of the four "
    "labels above, in plain language), (2) WHAT THE OFFICIAL METRIC SAYS, (3) WHAT THE TRUE METRIC SAYS, "
    "(4) COST IMPACT, (5) RECOMMENDED ACTION. Only draw conclusions the evidence actually supports."
)


@dataclass
class ReleaseReviewReport:
    category: str
    verdict: str
    official_result: object
    true_result: object
    cost_impact: dict
    narrative: str = ""
    llm_meta: dict = field(default_factory=dict)


def review_category(tickets: pd.DataFrame, category: str, client: LLMClient) -> ReleaseReviewReport:
    official_result = containment_shift_test(tickets, category, metric="official")
    true_result = containment_shift_test(tickets, category, metric="true")
    verdict = classify_policy_change(official_result, true_result)
    cost_impact = reopen_cost_impact(tickets, category)

    evidence = (
        f"Category: {category}\n"
        f"Rubric verdict: {verdict}\n\n"
        f"Official containment test (v1 vs v2): {vars(official_result)}\n\n"
        f"True (reopen-adjusted) containment test (v1 vs v2): {vars(true_result)}\n\n"
        f"Cost impact detail (v1 vs v2): {cost_impact}\n"
    )
    resp = client.complete(prompt=evidence, system=SYSTEM_PROMPT, prompt_id=f"release_review_{category}")

    return ReleaseReviewReport(
        category=category, verdict=verdict, official_result=official_result, true_result=true_result,
        cost_impact=cost_impact, narrative=resp.text, llm_meta={"backend": resp.backend, "cached": resp.cached},
    )
