# NorthStar Analytics — Retention, Funnel & Growth Metrics Command Center

**One-line pitch:** four connected growth-analytics techniques — cohort retention (real SQL, window functions), activation-funnel diagnosis, a data-driven North Star metric selection, and Kaplan-Meier survival analysis — applied to the same 4,000-account Flowdesk signup cohort, all independently surfacing the same underlying lever: Guided Setup onboarding measurably improves activation, retention, *and* survival.

## Why four techniques on one dataset, not four unrelated demos

A real product analytics role rarely lets you pick one technique and stop. A retention dip shows up in a cohort curve, gets diagnosed step-by-step in the activation funnel, gets checked against the right leading-indicator metric before anyone commits to it as a North Star, and gets quantified properly — with censored (still-active) accounts handled correctly, not discarded or mislabeled — in a survival model. This project is built so all four techniques measure the *same underlying signal* (a single latent "engagement propensity" drives every downstream metric in the data generator), so the story is genuinely coherent across all four lenses rather than four independently-drawn coincidences.

## What's actually built here (not just described)

- **`data/generate_data.py`** — a 4,000-account, 12-month Flowdesk signup cohort with an internally consistent generative model: one latent engagement-propensity variable drives activation-funnel behavior, North Star candidate metrics, weekly churn hazard, and current-state RFM features together, plus a long-format `weekly_activity` log (81,475 rows) purpose-built for a real SQL cohort-retention query.
- **`sql/01_cohort_retention.sql`** — a genuine window-function cohort-retention query: recovers each account's observed horizon via `MAX(week_since_signup)`, then restricts each milestone week's denominator to cohorts mature enough to have actually reached it, avoiding the common mistake of letting an immature cohort fake a retention drop.
- **`sql/02_activation_funnel.sql`** — step-by-step funnel conversion rates by onboarding path, isolating exactly where the biggest drop-off happens.
- **`pipeline/north_star.py`** — a "magic number" style North Star metric evaluation: restricts to cohorts mature enough to have an observed week-12 outcome (avoiding immaturity bias), then ranks candidate early-usage metrics by ROC-AUC against that outcome — not by which one sounds the most like "engagement."
- **`pipeline/segmentation.py`** — unsupervised k-means engagement segmentation (recency/frequency/breadth) on currently-active accounts, with segment labels generated dynamically from real cluster centroid values rather than a fixed rule-based tier system.
- **`pipeline/survival.py`** — Kaplan-Meier survival curves and a log-rank test (via `lifelines`), comparing time-to-churn by onboarding path — a genuinely different statistical tool from a binary churn classifier, because it uses censored (still-active) accounts correctly instead of discarding them.
- **`notebook/analysis.ipynb`** — the full, executed walkthrough of all five sections. Zero execution errors.
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards and a 4-panel Plotly view (cohort heatmap, activation funnel, North Star candidate ranking, survival curves).

## Key results

| Analysis | Result |
|---|---|
| Cohort retention | W1 89.2% → W4 68.7% → W8 50.8% → W12 39.1% → W16 30.6% (blended, maturity-adjusted) |
| Activation funnel | Guided Setup: 53.0% activation vs. Self-Serve: 36.4% (+17pp) — the gap is concentrated at the "core action" step, not first login |
| North Star metric | `week2_core_actions` recommended (AUC 0.65 vs. week-12 retention); `workspaces_created_wk1` rejected (AUC 0.51 — essentially random) |
| Engagement segments | Power User (173) / Regular (347) / Casual (424) / At-Risk-Active-but-Fading (287), from unsupervised k-means on currently-active accounts |
| Survival analysis | Median survival: Guided Setup 12 weeks vs. Self-Serve 8 weeks (log-rank p=6.2×10⁻¹⁷) |

## Resume bullets

- Built a real SQL cohort-retention pipeline using window functions to correctly handle cohort-maturity bias, and diagnosed a 17-point activation gap to its specific root cause (a single funnel step) via a step-by-step conversion-rate breakdown.
- Selected a North Star metric using a validated "magic number" methodology (ROC-AUC of candidate early-usage metrics against a maturity-adjusted week-12 retention outcome) rather than intuition, correctly rejecting an intuitively-plausible but non-predictive candidate metric (AUC 0.51).
- Applied Kaplan-Meier survival analysis (`lifelines`) with a log-rank significance test to quantify a 4-week median-survival gap between onboarding paths, correctly incorporating censored (still-active) accounts rather than discarding or mislabeling them — distinct from, and complementary to, classification-based churn scoring.
- Ran unsupervised k-means engagement segmentation on currently-active accounts, surfacing a 287-account "Active but Fading" segment invisible to a simple active/inactive status field.

## Interview talking points

- **"Why not just look at overall retention and call it a day?"** — an aggregate retention number hides *where* the problem is; the funnel breakdown localizes the Guided Setup advantage to one specific step (core action, not login), which is what makes the finding actionable rather than just descriptive.
- **"How did you avoid picking a North Star metric that just sounds good?"** — every candidate was scored against an actual outcome (week-12 retention, restricted to mature cohorts to avoid bias) using ROC-AUC, not correlation alone (which can mislead for binary outcomes) — and one intuitively reasonable candidate (`workspaces_created_wk1`) was correctly rejected because it doesn't predict anything.
- **"Why survival analysis instead of just a churn rate at 90 days?"** — a survival model correctly uses every account's data, including accounts that haven't churned yet (censored), rather than either discarding them or wrongly counting them as retained; it also gives a full time-to-event curve, so you can see *when* the risk is highest, not just a single snapshot number.
- **"What would you change with more time/data?"** — extend the North Star analysis beyond univariate candidates to a multivariate model (does combining `week2_core_actions` with a breadth-of-usage signal predict better than either alone?), and validate the segmentation's stability across multiple k-means random seeds before treating segment membership as operationally reliable.
