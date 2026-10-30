# ShipCall AI — Autonomous Experiment Readout & Ship/Hold/Iterate/Kill Decision Agent

**One-line pitch:** an agent that reviews a completed A/B test and reaches one of four decisions — SHIP, HOLD, ITERATE, or KILL — by running a fixed diagnostic playbook (Sample Ratio Mismatch check, guardrail-metric checks, and a formal heterogeneous-treatment-effect test) *before* an LLM ever synthesizes a recommendation, then applying an explicit decision rubric so the model's job is to apply the rule correctly, not invent one.

## Why this project, and why "primary metric went up" isn't a decision process

Five experiments are reviewed here, deliberately built so that judging each one by "is the primary metric significantly positive?" alone gets **3 out of 5 wrong**:

| Experiment | Primary metric alone says... | Correct decision | Why it's different |
|---|---|---|---|
| Onboarding Redesign v2 | Ship (significant, positive) | **SHIP** | Actually agrees — clean case |
| Aggressive Upsell Banner | Ship (significant, positive) | **HOLD** | 7-day churn guardrail moved significantly worse |
| New Pricing Page Layout | Kill (not significant) | **ITERATE** | Flat average hides a validated Enterprise win and Starter loss |
| Search Ranking Algorithm v3 | Ship (significant, positive) | **INVALID** | Sample Ratio Mismatch — the randomization pipeline is broken, so nothing here is interpretable |
| Dashboard Load Speed Opt | Kill (not significant) | **KILL** | Actually agrees — clean case |

This is the project's whole argument in one table: an experimentation program that reads the primary metric's p-value and stops is going to hold a party for a losing pricing test's fake win (Search Ranking), throw away a real segment-specific opportunity (Pricing Page), and ship something that quietly hurts retention (Upsell Banner).

## What's actually built here (not just described)

- **`data/generate_data.py`** — 5 independent, purpose-built experiment "results packages" (primary metric, 2 guardrail metrics, and a 3-segment breakdown each), each engineered to require a different one of the 5 decision outcomes — verified numerically after generation, not assumed.
- **`pipeline/srm.py`** — a Sample Ratio Mismatch check (chi-square goodness-of-fit test against the intended split), using the industry-standard stricter threshold (α=0.001, not 0.05) specifically because even healthy randomization occasionally produces a marginal p-value by chance.
- **`pipeline/guardrails.py`** — checks each guardrail metric with a two-proportion z-test, flagging a violation only when the metric moved *significantly* AND in the *harmful* direction (a guardrail moving favorably, even if "significant," is not a violation).
- **`pipeline/hte.py`** — a proper heterogeneous-treatment-effect test: reconstructs individual-level rows from aggregated segment counts and fits a single logistic regression with `treatment * segment` interaction terms, so "does the effect differ by segment" is a formal hypothesis test, not eyeballing whether point estimates look different (which inflates false-discovery risk through implicit multiple comparisons).
- **`pipeline/agent.py`** — the perceive → act → reason → report orchestration (`review_experiment()`), with an explicit 5-rule decision rubric given directly to the LLM in the system prompt, applied in strict priority order (SRM overrides everything; a guardrail violation overrides a positive primary result; HTE overrides a flat average).
- **`notebook/analysis.ipynb`** — the full, executed review of all 5 experiments, including the "primary metric alone would mislead on 3/5" summary chart. Zero execution errors.
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards, a Plotly overview chart, and all 5 full decision memos rendered as cards with their supporting evidence tables.

## On the LLM backend

This sandbox has no live LLM API access. `shared/llm_client.py` implements the same `LLMClient` abstraction used across this portfolio — `CachedClient` (replays real, pre-authored JSON responses, what this build uses) and `OpenAIClient`/`AnthropicClient` (drop-in live backends). Every cached decision memo was authored by reading the *actual* diagnostic output (real SRM chi-square statistics, real guardrail z-tests, real HTE interaction p-values) this notebook computes — not invented — and the decision rubric is explicit enough that a live model given the same evidence and system prompt would be expected to reach the same recommendation.

## Resume bullets

- Built an autonomous experiment-review agent combining a Sample Ratio Mismatch check, automated guardrail-metric monitoring, and a formal heterogeneous-treatment-effect test into a fixed diagnostic playbook, driving an explicit Ship/Hold/Iterate/Kill decision rubric that correctly overrode a naive "primary metric is significant" read on 3 of 5 reviewed experiments.
- Implemented heterogeneous treatment effect testing via interaction-term logistic regression (not per-segment eyeballing), correctly identifying a statistically validated Enterprise-segment win hidden inside a flat overall average.
- Implemented a Sample Ratio Mismatch detector using the industry-standard strict significance threshold (α=0.001), catching a broken randomization pipeline that would otherwise have shipped a spurious "winning" result (chi2=81.0, p=2.3×10⁻¹⁹).

## Interview talking points

- **"Why check SRM before anything else?"** — because it invalidates every other conclusion: if the two arms aren't comparable populations, a "significant" primary-metric difference can be entirely explained by whatever caused the imbalance, not by the treatment — so it has to be ruled out first, not last.
- **"Why interaction-term regression instead of just comparing each segment's numbers by eye?"** — comparing several segments' point estimates informally is an implicit multiple-comparisons problem (with enough segments, one will look different by chance); a single regression with formal interaction terms tests the actual claim ("does treatment effect differ by segment") with one well-defined p-value per comparison.
- **"Isn't a guardrail-triggered HOLD just being overly cautious?"** — the guardrail violation here (day7_churn +2.32pp on an 8.8% base, a ~26% relative increase) is proportionally larger than the primary metric's own lift (~12% relative) — this isn't marginal caution, it's a plausible net-negative-for-revenue outcome that a primary-metric-only view would ship blind to.
- **"What would you build next?"** — extending the guardrail check to support continuous (not just binary) metrics via CUPED-adjusted t-tests (reusing the companion CausalLift project's implementation), and letting the agent flag when an HTE finding is promising enough to warrant designing a dedicated segment-specific follow-up experiment rather than just recommending one in prose.
