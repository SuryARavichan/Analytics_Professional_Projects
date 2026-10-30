# GroundTruth Clinical — RAG Retrieval Quality & Clinical Hallucination Risk Analytics

**One-line pitch:** a RAG (retrieval-augmented-generation) quality analytics system for a clinical documentation AI copilot that catches a patient-safety risk hidden inside an improving category average — a rare-drug-combination retrieval regression (precision@k 0.44 → 0.28, p=6.6×10⁻⁴³) invisible in the drug_interaction category's overall numbers, because the same release improves the 80% of volume that's common drug combinations.

*Disclaimer: all clinical content in this project (guidelines, dosing, interactions) is synthetic and illustrative — built for a data-analytics portfolio, not real medical guidance.*

## Why this project, and why it's not "just another quality dashboard"

RAG is the dominant architecture for AI products right now specifically because retrieving real source documents before generating an answer is supposed to reduce hallucination. That claim needs its own verification — and the metrics that verify it are different from a general quality score: precision@k, recall@k, and MRR evaluate whether the RIGHT documents were retrieved, which is what lets you localize a failure to "the retrieval step got worse" instead of a vague "the answers got worse." This project builds those metrics directly, evaluates a real embedding-model upgrade with them, and finds the segment where retrieval quality got meaningfully worse while the category it belongs to looked fine.

## What's actually built here (not just described)

- **`data/generate_data.py`** — 24,000 synthetic clinical queries for Meridian Health AI across 4 query types and an embedding-model v1→v2 upgrade, with a deliberately embedded segment-level regression: `drug_interaction` queries about rare drug combinations (~20% of that category's volume) get worse retrieval post-v2 even as the common-combination majority improves — plus **16 real, hand-authored (query, retrieved-context, generated-answer) examples**, including two deliberately fabricated clinical claims in the exact regressing segment.
- **`pipeline/retrieval_metrics.py`** — precision@k, recall@k, and MRR computed per segment, plus a t-test/two-proportion-test toolkit that can test both a category aggregate and any sub-segment with the same functions.
- **`pipeline/hallucination_risk.py`** — hallucination rate AND **silent failure rate** (hallucination occurred *and* the physician accepted the answer as-is) — the metric that actually matters for patient safety, since a caught hallucination is a near-miss and an uncaught one is a real error reaching a clinical decision.
- **`pipeline/judge_validation.py`** — the expensive path: an LLM judge scores real text examples against a strict groundedness rubric (every claim must be traceable to the retrieved context, full stop, regardless of plausibility).
- **`pipeline/agent.py`** — the perceive → act → reason → report loop: for each segment, an LLM applies an explicit rubric that forbids excusing a segment-level regression with a healthy category aggregate.
- **`notebook/analysis.ipynb`** — the full, executed walkthrough: retrieval metrics by segment, the category-aggregate-vs-segment comparison, safety metrics, proxy-vs-judge validation, and all five segment release-review reports. Zero execution errors (22 cells).
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards, a 4-panel Plotly view, the two real hallucinated examples as evidence cards, and all five segment verdicts.

## Key results

| Segment | Precision@k (v1→v2) | p-value | Hallucination rate (v1→v2) | Verdict |
|---|---|---|---|---|
| medication_dosage | 0.585 → 0.641 | 1.3×10⁻²⁸ | 18.5% → 15.2% | Improved |
| diagnosis_criteria | 0.487 → 0.526 | 1.9×10⁻⁹ | 22.3% → 19.3% | Improved |
| care_protocol | 0.630 → 0.672 | 9.6×10⁻¹² | 13.8% → 14.3% (n.s.) | Improved |
| drug_interaction (common, 80% of volume) | 0.477 → 0.522 | 4.2×10⁻¹² | 20.7% → 19.8% (n.s.) | Improved |
| **drug_interaction (rare, ~20% of volume)** | **0.440 → 0.283** | **6.6×10⁻⁴³** | **23.3% → 28.7%** (p=0.028) | **Patient safety risk** |
| drug_interaction (category aggregate) | 0.470 → 0.472 | — | 21.2% → 21.7% | *Looks fine* |

Proxy-vs-judge groundedness correlation: **r=0.74** (n=10 segment×version groups), anchored largely by the drug_interaction_rare/v2 point where both the cheap proxy and the expensive judge independently show a collapse.

## On the honest limitations (stated in the notebook, not hidden)

- **This is a simulated retrieval evaluation, not a real document-corpus benchmark.** Retrieval outcomes are generated from a per-segment "quality probability" plus a ground-truth relevant-chunk count, not an actual embedding space and document corpus — enough to demonstrate the metrics and the finding pattern, not a substitute for evaluating against a real labeled corpus.
- **The proxy-validation correlation (r=0.74, n=10) is directionally strong but not a tight estimate at this sample size**, and it's meaningfully driven by one extreme point — the notebook says so explicitly rather than overselling the number.
- **All clinical content is synthetic and illustrative.** The specific interactions, dosing thresholds, and guideline citations are invented for this dataset and must never be read as real medical guidance.

## Resume bullets

- Built a RAG-quality analytics pipeline (precision@k, recall@k, MRR) for a clinical AI copilot that detected a retrieval-quality regression (0.440 → 0.283 precision@k, p=6.6×10⁻⁴³) in a specific query segment, completely hidden inside a category-level aggregate that looked stable-to-improving across the same release.
- Defined and measured "silent failure rate" (a hallucination that occurs AND is accepted without review) as the patient-safety-relevant outcome metric, distinct from raw hallucination rate, and showed it rising specifically in the segment where human reviewers were least equipped to catch an error.
- Validated a cheap, always-on citation-coverage proxy against an expensive LLM-judge groundedness score on real hand-authored clinical examples (r=0.74, n=10 groups), demonstrating the proxy would have flagged the same at-risk segment without waiting for a manual audit.

## Interview talking points

- **"Why evaluate retrieval separately from the final answer?"** — because it localizes the failure. If the final answer is bad but retrieval was good, the problem is generation/grounding; if retrieval itself is bad, no amount of prompting fixes it — you need better retrieval. Precision@k/recall@k/MRR are what make that distinction possible instead of guessing.
- **"How is 'silent failure rate' different from just 'hallucination rate'?"** — hallucination rate tells you how often the model gets it wrong; silent failure rate tells you how often that wrong answer actually reaches a clinical decision unchecked. The two can move independently — this dataset's rare-combination segment shows both rising together, which is the worst combination: more errors, and less chance any given error gets caught.
- **"Why not just look at the drug_interaction category number?"** — because it's the aggregate that's misleading here: the 80%-of-volume common-combination majority improving is enough to keep the category average looking fine while a real, large, statistically unambiguous regression happens in the other 20%. This is the general lesson: any category-level AI-quality metric needs a segment-level check before you trust it, especially in a domain where a missed segment carries real consequences.
- **"What would you change with more time/data?"** — evaluate against an actual labeled document corpus rather than a simulated retrieval-quality probability, and expand the judge-sample size well beyond 16 to tighten the proxy-validation correlation's confidence interval.
