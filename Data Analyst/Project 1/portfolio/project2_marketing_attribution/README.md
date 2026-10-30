# Project 2 — Checkout A/B Test & Multi-Touch Marketing Attribution

**One-line pitch:** Ran a statistically rigorous read on an A/B test
(significance + power + guardrails + segment cuts), then modeled how the
choice of attribution model changes marketing budget decisions — backed by
an interactive Excel what-if model.

## Business problem

Two linked questions a growth/marketing team actually asks:

1. We shipped a new checkout flow and conversion went up — can we trust
   that, was the test big enough to mean anything, and did we break
   anything else in the process?
2. Which marketing channels are actually working? Last-touch attribution
   says Paid Social is a laggard — is that true, or is it doing invisible
   work earlier in the funnel?

## Data

- `data/ab_test_sessions.csv` — 48,000 sessions from a synthetic 30-day
  randomized checkout A/B test (device, traffic source, conversion, order
  value, page-load-time guardrail).
- `data/touchpoints.csv` — 9,000 synthetic multi-touch marketing journeys
  (1-5 touchpoints across 5 channels) with conversion outcome/value.
- `data/channel_spend.csv` — 6 months of monthly spend/clicks per channel.

Regenerate with `cd data && python3 generate_data.py` (writes CSVs +
`marketing.db` SQLite database).

## Method

| Step | Tool | File |
|---|---|---|
| Topline + device-segment test results | SQL | `sql/01_ab_test_results.sql` |
| Multi-touch attribution (first/last/linear/position-based via window functions) | SQL | `sql/02_attribution_models.sql` |
| Channel ROAS under linear attribution | SQL | `sql/03_channel_roas.sql` |
| Two-proportion z-test, Wilson confidence intervals, retrospective power analysis, guardrail t-test, segment cut, attribution comparison charts | Python (`scipy`, `statsmodels`) | `notebook/analysis.ipynb` |
| Interactive budget-reallocation what-if model | Excel (live formulas) | `excel/budget_reallocation_model.xlsx` |
| Stakeholder dashboard | Python (Plotly) | `dashboard/index.html` — **open in any browser** |

## Key findings

**A/B test**
- Overall conversion lift: **+1.53pp** (10.52% treatment vs. 9.00% control),
  z=5.64, p<0.0001 — statistically significant, and the test was
  **over-powered** (achieved power ≈1.00 vs. an 80% target, with adequate
  sample size at ~5,900/group).
- Guardrail check: page load time rose a statistically detectable but
  practically acceptable +36ms — ship it, but keep monitoring.
- **The lift is almost entirely a Mobile effect**: +2.67pp on Mobile
  (p<0.0001) vs. **not significant** on Desktop (-0.37pp, p=0.79). A
  topline-only read would have missed this and risked a lower-value
  blanket rollout narrative instead of a targeted one.

**Attribution**
- Attribution model choice changes the "best channel" answer. Paid Social
  ranks last under last-touch (it rarely closes) but 2nd under first-touch
  (it's a strong early-funnel driver) — its true contribution is invisible
  to a last-touch-only view.
- Under linear attribution, Paid Search and Paid Social are running
  **below 1.0x ROAS** (0.22x and 0.17x) despite being the two biggest
  budget lines, while Organic, Email, and Referral are all solidly
  profitable (1.5x–2.5x).

## Recommendations delivered to "leadership"

1. Roll out the checkout redesign to Mobile now; hold Desktop for a
   follow-up test rather than crediting Desktop with a lift it didn't
   produce.
2. Stop optimizing budget off last-touch ROAS alone; use linear or
   position-based attribution for budget planning and reserve last-touch
   for real-time bid optimization only.
3. Shift budget from the two underperforming paid channels toward Organic/
   Email/Referral. `excel/budget_reallocation_model.xlsx` models a
   budget-neutral -10%/-10%/+61.1%/+61.1%/+61.1% shift (with a diminishing-
   returns elasticity assumption on the funded channels) projecting a
   **+$46.6k revenue impact** over the same period, at zero net spend
   change — edit the yellow input cells to test other splits.

## How to talk about this in an interview

- "I didn't stop at 'the lift is significant' — I ran a power analysis to
  show the test could actually detect an effect of that size, which is the
  check most people skip."
- "I caught a Simpson's-paradox-adjacent pattern by segmenting before
  trusting the blended number — that's the difference between a topline
  metric and an actionable recommendation."
- "I can explain, concretely, why last-touch attribution systematically
  undervalues upper-funnel channels, and I built a model that lets a
  stakeholder test reallocation scenarios themselves instead of trusting a
  single static recommendation."
