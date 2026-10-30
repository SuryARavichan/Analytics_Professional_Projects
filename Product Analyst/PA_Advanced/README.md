# Advanced Product Analyst Portfolio — Experimentation, Growth & AI Decision Agents

Three projects built around Flowdesk (the same fictional B2B SaaS product
used across this session's portfolios), each demonstrating a distinct
advanced skill set that separates a senior/PM-track product analyst from
a junior data analyst: rigorous experimentation and causal inference,
growth and retention analytics with a data-driven North Star metric, and
an AI-augmented decision agent that combines both into an automated
experiment-review process.

| # | Project | The advanced PA skill it demonstrates |
|---|---|---|
| 1 | [**CausalLift**](1_causallift_experimentation/) — Statistically Rigorous A/B Testing & Causal Impact Engine | CUPED variance reduction, a Monte-Carlo-calibrated fix for the "peeking problem" (quantified at a 5x false-positive inflation), and difference-in-differences for a rollout that couldn't be randomized |
| 2 | [**NorthStar Analytics**](2_northstar_growth_retention/) — Retention, Funnel & Growth Metrics Command Center | Real SQL cohort retention, activation-funnel diagnosis, a validated (not guessed) North Star metric selection, unsupervised engagement segmentation, and Kaplan-Meier survival analysis |
| 3 | [**ShipCall AI**](3_shipcall_ai_decision_agent/) — Autonomous Experiment Readout & Ship/Hold/Iterate/Kill Decision Agent | Sample Ratio Mismatch detection, automated guardrail-metric monitoring, formal heterogeneous-treatment-effect testing, and an LLM decision agent that correctly overrides a naive "primary metric is positive" read on 3 of 5 experiments |

Each project's `dashboard/index.html` is self-contained — open it directly
in a browser, no server or install needed.

## Why these three, together

They're deliberately not three versions of the same A/B test. CausalLift
is the *rigor* layer — the specific, well-documented ways experimentation
programs get it wrong (ignoring variance reduction, the peeking problem,
treating a non-randomized rollout as causal) and their standard fixes.
NorthStar Analytics is the *growth-metrics* layer — the day-to-day
toolkit of cohort retention, funnel diagnosis, metric selection, and
survival analysis that most product analytics roles actually run week to
week. ShipCall AI is the *synthesis* layer — an AI agent that combines
disciplined experimentation checks (SRM, guardrails, HTE) with an
explicit decision rubric, showing that "AI-augmented" analytics means
guardrailing and structuring a model's judgment, not replacing statistical
rigor with it.

## An important design decision, upfront

ShipCall AI (project 3) uses an LLM to generate its decision-memo
narratives. This sandbox has no live LLM API access, so — consistent with
every AI-augmented project across this session's portfolios —
`shared/llm_client.py` implements an `LLMClient` abstraction with
interchangeable backends: `CachedClient` (the default; replays
pre-authored responses from a local JSON fixture, every one of them
written by reading the *actual* diagnostic output the notebook computes)
and `OpenAIClient`/`AnthropicClient` (drop-in live backends, auto-selected
when an API key is present). CausalLift and NorthStar Analytics use no
LLM at all — every result in those two projects is a directly computed
statistic, which is itself worth noting: the majority of "advanced"
product-analyst work is disciplined statistics, not AI.

## Key numbers across the portfolio

- **CausalLift**: naive daily-peeking A/B monitoring produces a 24.9%
  true false-positive rate against a 5% nominal target (5,000-simulation
  Monte Carlo); a calibrated sequential boundary brings it to 5.1% and
  correctly withholds significance on a real seeded false-positive
  instance. CUPED reduces outcome variance 44.4%. A staged feature
  rollout's causal effect is estimated at +4.45pp (95% CI: 4.18pp to
  4.72pp) via difference-in-differences, validated with a parallel-trends
  check and a clean placebo test.
- **NorthStar Analytics**: Guided Setup onboarding drives a +17pp
  activation lift, a materially better cohort retention curve, the
  dominant signal behind the recommended North Star metric
  (`week2_core_actions`, AUC 0.65 vs. an intuitively-plausible but
  non-predictive alternative at AUC 0.51), and a 12-week vs. 8-week
  median survival time (log-rank p=6.2×10⁻¹⁷).
- **ShipCall AI**: 5 experiments reviewed; a primary-metric-only read
  would reach the wrong decision on 3 of them (a guardrail-violating
  "win," a flat average hiding a real segment-specific win, and a
  seemingly-significant result invalidated by a Sample Ratio Mismatch,
  chi2=81.0, p=2.3×10⁻¹⁹).

## Resume bullet points

- Quantified the false-positive-rate inflation from naive A/B-test
  peeking via a 5,000-simulation Monte Carlo study and implemented a
  Monte-Carlo-calibrated sequential testing boundary to correct it;
  implemented CUPED variance reduction from scratch (44% variance
  reduction on a real experiment).
- Built a difference-in-differences causal-impact pipeline (two-way fixed
  effects, cluster-robust SEs, parallel-trends check, placebo test) for a
  non-randomized staged feature rollout, correctly identifying and
  validating a +4.45pp causal effect.
- Built a real SQL cohort-retention pipeline (window functions,
  maturity-adjusted denominators), diagnosed a 17-point activation gap to
  its specific funnel step, and selected a North Star metric using a
  validated "magic number" methodology (ROC-AUC against a real retention
  outcome) rather than intuition.
- Applied Kaplan-Meier survival analysis with a log-rank test to quantify
  a churn-timing gap between onboarding cohorts, correctly incorporating
  censored (still-active) accounts.
- Designed and built an autonomous experiment-decision agent combining
  Sample Ratio Mismatch detection, automated guardrail monitoring, and
  formal heterogeneous-treatment-effect testing with an LLM applying an
  explicit decision rubric — correctly overriding a naive primary-metric
  read on 3 of 5 reviewed experiments.

## Interview prep — the questions this portfolio invites

- *"Why does the peeking problem matter if you're just going to look at
  the final result anyway?"* — because in practice, teams don't wait;
  CausalLift's own Monte Carlo simulation shows a ~5x true-error-rate
  inflation from informal daily monitoring, and a real, undoctored
  instance where a true-null experiment would have been shipped as a
  winner on day 8 without the correction.
- *"How do you know your causal estimate (DiD) isn't just correlation?"*
  — the parallel-trends check (pre-period slopes for treated vs. control
  regions are close) and the placebo test (a fake mid-pre-period
  "rollout" produces no effect, p=0.96) are both explicit, falsifiable
  checks of the identifying assumption, not just running the regression
  and reporting the coefficient.
- *"How did you pick the North Star metric, and how do you know it's not
  just a guess with a good name?"* — every candidate was scored by
  ROC-AUC against an actual maturity-adjusted retention outcome, and one
  intuitively reasonable candidate was correctly rejected for having
  essentially no predictive power (AUC 0.51).
- *"Isn't an LLM making ship/kill decisions risky?"* — ShipCall AI's
  design is the answer: the LLM never touches raw data or invents
  statistics — it synthesizes the output of a fixed, deterministic
  diagnostic playbook (SRM, guardrails, HTE) into a memo, following an
  explicit rubric given directly in the system prompt, which is what
  makes 3 of its 5 "surprising" recommendations traceable to a specific,
  auditable number rather than to model judgment.

## Requirements to re-run anything

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn \
            plotly nbformat jupyter lifelines openai anthropic
```

`lifelines` (used in NorthStar Analytics' survival analysis) can be
finicky to build against a system Python's `setuptools`; if `pip install`
fails on `autograd-gamma`, use a clean virtual environment
(`python3 -m venv venv && venv/bin/pip install lifelines ...`) instead.

Each project's `data/generate_data.py` is self-contained and reproducible
(seeded random generators). To rebuild a notebook: `cd notebook &&
python3 build_notebook.py && jupyter nbconvert --to notebook --execute
--inplace analysis.ipynb`. To rebuild a dashboard: `cd dashboard &&
python3 build_dashboard.py`. To go live with a real LLM in ShipCall AI:
`export OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`) before running any
notebook — no code changes needed.
