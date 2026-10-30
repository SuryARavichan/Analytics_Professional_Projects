"""
agent.py — orchestrates the "perceive -> act -> reason -> report" loop for
experiment decision-making:

  PERCEIVE  a completed experiment's primary-metric result
  ACT       run a fixed diagnostic playbook (srm.py, guardrails.py,
            hte.py) -- deterministic, real statistics, always run the
            same way for every experiment, and run BEFORE the primary
            result is ever trusted (an SRM failure invalidates everything
            else, which is why it's checked first)
  REASON    an LLM call synthesizes the diagnostic evidence into a
            structured Ship / Hold / Iterate / Kill recommendation
            following an explicit decision rubric, for a cross-functional
            experiment-review audience (PM, eng, data)
  REPORT    a structured DecisionMemo the dashboard/notebook renders

SCOPING NOTE: this is a fixed diagnostic playbook triggered on every
experiment read, not a dynamic tool-selection loop -- the same scope
decision made in the companion LaunchPulse project, for the same reason:
every experiment gets identical, auditable evidence-gathering, so the
recommendation doesn't depend on the model's tool-choice quality.

DECISION RUBRIC (given to the model explicitly, not left to guesswork):
  1. If SRM is detected -> INVALID. Do not interpret any other result;
     recommend fixing the bucketing pipeline and re-running.
  2. Else if any guardrail is violated -> HOLD. Even a significant,
     positive primary-metric result should not ship if it comes with a
     significant guardrail-metric cost.
  3. Else if a significant heterogeneous treatment effect exists across
     segments -> ITERATE. Recommend a segment-specific rollout (or
     further investigation) rather than a blanket ship/kill.
  4. Else if the primary metric is significant and positive -> SHIP.
  5. Else -> KILL. No detected effect and nothing else to salvage.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from llm_client import LLMClient  # noqa: E402
from srm import check_srm  # noqa: E402
from guardrails import check_all_guardrails  # noqa: E402
from hte import test_heterogeneous_effects  # noqa: E402
from ab_testing_lite import two_proportion_summary  # noqa: E402

SYSTEM_PROMPT = (
    "You are ShipCall AI, an autonomous experiment-review agent for Flowdesk, a B2B SaaS product. "
    "You are given one completed experiment's primary-metric result and the output of a fixed "
    "diagnostic playbook: a Sample Ratio Mismatch (SRM) check, guardrail-metric checks, and a "
    "heterogeneous-treatment-effect (HTE) test across customer segments. Apply this decision rubric "
    "in strict order and state which rule you applied: "
    "(1) If SRM is detected, the recommendation MUST be INVALID -- do not interpret the primary "
    "result at all, and say so explicitly. "
    "(2) Else if any guardrail is violated, the recommendation MUST be HOLD, even if the primary "
    "metric is significantly positive. "
    "(3) Else if a significant heterogeneous treatment effect exists across segments, the "
    "recommendation MUST be ITERATE, and you should describe the specific segment-level pattern "
    "and a concrete segment-specific next step. "
    "(4) Else if the primary metric is significant and positive, the recommendation MUST be SHIP. "
    "(5) Otherwise, the recommendation MUST be KILL. "
    "Write a short decision memo with: (1) RECOMMENDATION (one of INVALID/HOLD/ITERATE/SHIP/KILL), "
    "(2) EVIDENCE (the specific numbers driving the decision), (3) RATIONALE (why this rule applied, "
    "not a different one), (4) NEXT STEP. Only draw conclusions the evidence actually supports."
)


@dataclass
class DecisionMemo:
    experiment_id: str
    name: str
    srm: object
    guardrail_results: list
    hte: object
    primary_result: dict
    narrative: str = ""
    llm_meta: dict = field(default_factory=dict)


def review_experiment(experiment: dict, client: LLMClient) -> DecisionMemo:
    # ACT: fixed diagnostic playbook, always in this order
    srm_result = check_srm(experiment["control_n"], experiment["treatment_n"], experiment.get("intended_split", 0.5))
    guardrail_results = check_all_guardrails(experiment["guardrails"])
    hte_result = test_heterogeneous_effects(experiment["segments"], reference_segment="Starter")
    primary = two_proportion_summary(experiment["control_n"], experiment["control_conversions"],
                                      experiment["treatment_n"], experiment["treatment_conversions"])

    evidence = (
        f"Experiment: {experiment['name']}\n"
        f"Hypothesis: {experiment['hypothesis']}\n"
        f"Primary metric: {experiment['primary_metric']}\n\n"
        f"--- SRM check ---\n"
        f"Observed split: {srm_result.observed_ratio:.3f} (expected {srm_result.expected_ratio:.3f}), "
        f"chi2={srm_result.chi2:.2f}, p={srm_result.p_value:.4f}, SRM detected: {srm_result.is_srm}\n\n"
        f"--- Primary metric result ---\n"
        f"Control: {primary['p_control']:.3f}  Treatment: {primary['p_treatment']:.3f}  "
        f"Lift: {primary['lift_pp']:+.2f}pp  z={primary['z']:.2f}  p={primary['p_value']:.4f}\n\n"
        f"--- Guardrail checks ---\n"
        + "\n".join(f"{g.name}: control={g.control_rate:.3f} treatment={g.treatment_rate:.3f} "
                     f"delta={g.delta_pp:+.2f}pp p={g.p_value:.4f} VIOLATED={g.violated}" for g in guardrail_results)
        + "\n\n--- Heterogeneous treatment effect (by segment) ---\n"
        + hte_result.segment_effects.to_string(index=False)
        + f"\nInteraction term p-values: {hte_result.interaction_p_values}\n"
        f"Any significant interaction: {hte_result.any_significant_interaction}\n"
    )
    resp = client.complete(prompt=evidence, system=SYSTEM_PROMPT, prompt_id=experiment["experiment_id"])

    return DecisionMemo(
        experiment_id=experiment["experiment_id"], name=experiment["name"],
        srm=srm_result, guardrail_results=guardrail_results, hte=hte_result, primary_result=primary,
        narrative=resp.text, llm_meta={"backend": resp.backend, "cached": resp.cached},
    )
