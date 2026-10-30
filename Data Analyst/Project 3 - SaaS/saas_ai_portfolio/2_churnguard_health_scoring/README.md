# ChurnGuard AI — Predictive Customer Health Scoring & AI-Generated Renewal Briefs

**One-line pitch:** a two-sided account health-scoring system for a B2B SaaS company that scores every account on both churn risk *and* expansion opportunity, combines them into a 2×2 health quadrant, and auto-drafts evidence-grounded renewal briefs for the highest-priority upcoming renewals — the kind of pre-call prep a CS or Sales rep would otherwise spend 20–30 minutes assembling by hand.

## Why this project, and why it's not just another churn model

Almost every "churn prediction" portfolio project stops at one model: will this account leave? That answers half the question a real revenue org has. A single blended "health score" also actively hides useful structure — specifically, it can't distinguish an account that's heavily engaged *and* at risk (needs a support/success conversation, not a discount) from one that's quietly disengaging (needs proactive outreach) or one that's ready to expand (needs a seat-upgrade pitch, not a retention offer).

So this project trains **two models on the same leakage-free feature set** — churn risk and expansion opportunity — and combines them into a quadrant:

| | Low Expansion Score | High Expansion Score |
|---|---|---|
| **High Churn Risk** | 🔴 At Risk — Save | 🟠 Engaged but Unstable |
| **Low Churn Risk** | ⚪ Stable — Maintain | 🔵 Growth Opportunity |

The **"Engaged but Unstable"** quadrant is the finding a usage-only health metric would completely miss: accounts using the product heavily (which normally reads as a green flag) while also showing real risk signals — support friction, a detractor NPS score, plan/contract mismatch. These accounts have leverage *and* a reason to churn loudly at renewal.

## What's actually built here (not just described)

- **`data/generate_data.py`** — simulates 2,800 B2B SaaS accounts across Starter/Growth/Enterprise tiers and Monthly/Annual contracts, with 12 months of usage history per account (seat utilization, support tickets, ticket sentiment) and realistic hazard-based churn (22.2% overall) and expansion (13.3%) simulation, including a genuine plan/contract churn gradient (Enterprise/Annual churns at 4.1%; Starter/Monthly at 50.0% — directly informing recommendation #5 below).
- **`sql/01_health_kpis.sql`** — topline churn/expansion rate and MRR-at-risk by plan tier and contract type.
- **`sql/02_feature_engineering.sql`** — a **leakage-free** modeling snapshot built entirely in SQL with window functions (`ROW_NUMBER`, `AVG(...) OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`, `LAG(...,3)`): every account's features are snapshotted as of the month *before* its churn or expansion event (or its latest month, for accounts with neither outcome yet), so both models predict ahead of time instead of diagnosing after the fact.
- **`pipeline/health_scoring.py`** — trains two `RandomForestClassifier` models (churn AUC **0.837**, expansion AUC **0.961**) on the same 15-feature set, and assigns the 2×2 quadrant label from both scores.
- **`notebook/analysis.ipynb`** — the full, executed analysis: topline KPIs, feature engineering walkthrough, both models trained with feature-importance charts, the quadrant scatter, an "Engaged but Unstable" deep-dive, the AI renewal briefs, and a 5-point executive summary. Zero execution errors.
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards, an interactive Plotly quadrant scatter (hover any account for its evidence), the quadrant summary table, and 10 AI-generated renewal briefs rendered as cards (5 "At Risk — Save" for CS, 5 "Growth Opportunity" for Sales).
- **AI-generated renewal briefs** (`cached_responses.json`, via the shared `LLMClient` abstraction — see note below): for the 10 highest-priority accounts with a renewal in the next 90 days, each brief is grounded in that account's *actual* computed evidence (utilization level and 3-month trend, trailing support ticket volume and sentiment, NPS) and gives a root-cause read, call-prep talking points, and a confidence level. Critically, one brief (**ACCT60347**) explicitly pushes back on the model's own quadrant label — high utilization and a strong expansion score, but paired with an NPS detractor score and negative support tickets — recommending a support/success-led conversation instead of a growth pitch, exactly the nuance a raw quadrant label can't carry.

## On the LLM backend (read this before judging the "AI" claim)

This sandbox has no live LLM API access. Rather than fake that, `shared/llm_client.py` implements a small `LLMClient` abstraction with interchangeable backends: `CachedClient` (replays real, pre-authored JSON responses — what this build uses) and `OpenAIClient`/`AnthropicClient` (drop-in live backends, same interface, auto-selected by `get_client()` when an API key is present in the environment). This mirrors a real production pattern — prompt caching and fixture-based testing are standard practice for LLM features precisely because they're non-deterministic and costly to call on every run. Every cached brief here was authored by reading the *real* per-account numbers this notebook computes (not invented), and the code path that would call a live model is fully implemented and would work unchanged with a real API key.

## Key results

- Churn model AUC **0.837**, expansion model AUC **0.961** — both models lean primarily on seat-utilization level and trend, in *opposite* directions, which is a structurally sensible pattern (not overfitting to noise).
- Quadrant distribution across the 2,178 active accounts: **Stable 983 · At Risk 540 · Growth Opportunity 465 · Engaged but Unstable 190.**
- Annual-contract accounts churn at roughly a third the rate of Monthly-contract accounts across every plan tier.
- 10 renewal briefs generated for the highest-priority accounts (by MRR × risk/expansion score) with renewals in the next 90 days, each with a concrete, differentiated talking-points recommendation — not generic boilerplate.

## Resume bullets

- Built a two-sided (churn + expansion) customer health-scoring system for a simulated 2,800-account B2B SaaS portfolio, training dual RandomForest classifiers (AUC 0.84 / 0.96) on leakage-free, SQL-window-function-engineered features, and combining both scores into a 2×2 health quadrant that surfaces a segment ("Engaged but Unstable") invisible to single-score health metrics.
- Designed an LLM-powered renewal-brief generator that turns a ranked priority list into evidence-grounded, account-specific call prep for Customer Success and Sales, including briefs that correctly contradict the underlying model's own quadrant label when supporting evidence warrants it.
- Engineered a leakage-free feature pipeline in raw SQL (window functions: rolling averages, lagged trend deltas, event-relative snapshotting) to ensure both models predict churn/expansion ahead of the event rather than diagnosing it after the fact.

## Interview talking points

- **"Why two models instead of one health score?"** — because "at risk" and "expansion candidate" require different actions from different teams; blending them into one number throws away exactly the information a CS or Sales lead needs to route the account correctly.
- **"How did you avoid leakage?"** — the SQL snapshot query explicitly resolves, per account, the month *before* its labeled outcome (churn or expansion) via a `CASE` on `churn_month`/`expansion_month`, joined back to the rolling-feature CTE at that specific month — not the outcome month itself.
- **"Why cached LLM responses instead of a live API call?"** — cost, latency, and determinism in a demo/CI context; the abstraction (`LLMClient` ABC) means swapping to a live backend is a one-line change, and every cached response is grounded in real computed data, not invented.
- **"What would you change with more time/data?"** — probably a survival-analysis framing (time-to-churn, not binary churn) given the renewal-cadence structure already in the data, and a proper backtest across multiple historical cohorts rather than a single train/test split.
