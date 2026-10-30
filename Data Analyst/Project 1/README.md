# Data Analyst Portfolio — 3 Full-Stack Projects

Three end-to-end analytics projects built to demonstrate the full range a
data analyst role expects in 2026: SQL (schema design, CTEs, window
functions), Python (statistics, hypothesis testing, light ML), a live
Excel model, and interactive dashboards — each anchored to a realistic
business problem with a written recommendation, not just a chart.

Every project follows the same structure so a reviewer can navigate any of
them the same way:

```
projectN_.../
  data/            generator script + CSVs + a SQLite .db (re-runnable, inspectable)
  sql/             numbered .sql files, commented, using CTEs/window functions
  notebook/        analysis.ipynb — executed, with saved outputs/charts
  dashboard/       index.html — a self-contained interactive dashboard (open in any browser)
  excel/           (project 2 only) a live-formula what-if Excel model
  README.md        the case study: problem, method, findings, recommendations
```

## The three projects

| # | Project | Business problem | Tools showcased |
|---|---|---|---|
| 1 | [E-commerce Customer Retention & Cohort Analysis](project1_ecommerce_retention/) | Diagnose a retention problem, segment customers by value, trace a KPI dip to its operational root cause | SQL (cohorts, RFM via `NTILE()`), Python (chi-square test, CLV), dashboard |
| 2 | [Checkout A/B Test & Multi-Touch Attribution](project2_marketing_attribution/) | Validate an A/B test rigorously (significance + power + guardrails), then fix a flawed last-touch attribution model | SQL (attribution models via window functions), Python (`statsmodels` power analysis, z-tests), interactive Excel budget model, dashboard |
| 3 | [SaaS Churn Prediction & Diagnosis](project3_saas_churn/) | Predict churn before it happens (not after) and hand Customer Success a ranked, actionable account list | SQL (leakage-aware feature engineering), Python (`scikit-learn` logistic regression + random forest, AUC ≈0.86), dashboard |

Each project's dashboard (`dashboard/index.html`) is fully self-contained
(Plotly.js inlined) — double-click to open, or drag into a browser. No
server, no install.

## Why these three, together

They're deliberately not three versions of the same skill. Project 1 is
descriptive/diagnostic analytics (what happened, and why). Project 2 is
inferential statistics and decision-support (is this real, and what should
we do about it). Project 3 is predictive analytics (what will happen, and
who should act on it). That range — plus the fact that every project
includes SQL, Python, *and* a business write-up — is what a portfolio
needs to show in 2026: not just "I can make a chart," but "I can take a
business question all the way to a decision."

## Resume bullet points

Use/adapt whichever fit the role you're applying to:

- Built an end-to-end cohort retention analysis on 22K+ e-commerce orders
  using SQL window functions and Python, identifying a statistically
  significant (p<0.001) 2x loyalty gap between acquisition channels and
  tracing a Q3 retention dip to a specific operational incident.
- Designed and validated an A/B test analysis pipeline (two-proportion
  z-test, retrospective power analysis, guardrail metrics) on 48K sessions,
  catching a device-level heterogeneous treatment effect a topline read
  would have missed.
- Built a multi-touch attribution model in SQL comparing 4 credit-
  assignment methods, revealing that last-touch attribution undervalued a
  top-2 channel by over 40% — then built an interactive Excel what-if model
  projecting a $46K+ revenue impact from a proposed budget reallocation.
- Developed a leakage-free churn prediction model (logistic regression +
  random forest, ~0.86 AUC) from SQL-engineered rolling engagement
  features on 2,600 SaaS accounts, delivering a ranked, MRR-weighted
  at-risk account list for Customer Success outreach.
- Delivered 3 full-stack analytics projects (SQL, Python, Excel,
  interactive dashboards) each translating a business question into a
  concrete, quantified recommendation.

## Interview prep — the questions this portfolio invites

Be ready for these; they're the natural next question after "walk me
through one of your projects":

- *"Why SQLite instead of a real warehouse?"* — It's a portable, zero-
  setup stand-in for the same SQL a Postgres/Snowflake/BigQuery warehouse
  would run; every query here uses standard CTEs and window functions that
  port directly (SQLite's date functions are the only warehouse-specific
  bit).
- *"How do you know your model/test isn't just noise?"* — Point to the
  power analysis and significance tests in Project 2, and the train/test
  split + AUC in Project 3.
- *"What would you do differently with real data?"* — Real data would need
  more data-quality triage (duplicates, timezone bugs, currency
  conversion), and you'd want to validate the "story" against a domain
  expert rather than knowing it by construction, as you do here with
  synthetic data.
- *"Why synthetic data?"* — It lets the project demonstrate a specific,
  realistic business scenario end-to-end without a data-sharing/licensing
  problem, while every technique (the SQL, the stats, the model) is
  identical to what you'd run against real production data.

## Requirements to re-run anything

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn plotly openpyxl jupyter nbformat
```

Each project's `data/generate_data.py` is self-contained and reproducible
(seeded random generators) — re-run it to regenerate the CSVs/database from
scratch, then re-run `notebook/build_notebook.py` + `jupyter nbconvert
--execute` and `dashboard/build_dashboard.py` to rebuild the notebook and
dashboard.
