# Project 3 — SaaS Churn Prediction & Diagnosis

**One-line pitch:** Built a leakage-free churn prediction model (logistic
regression + random forest, ~0.86 AUC) from SQL-engineered rolling
engagement features, then turned it into a ranked, MRR-weighted account
list a Customer Success team could act on tomorrow.

## Business problem

A B2B SaaS company wants to know: what's driving churn, and can we flag
at-risk accounts *before* they cancel — early enough that Customer Success
can actually do something about it?

## Data

24 months (2024-01–2025-12) of synthetic subscription + usage data for
2,600 B2B accounts: `customers` (plan tier, contract type, region,
industry, NPS, churn outcome) and `usage_monthly` (logins, active users,
feature adoption, support tickets per account per month). NPS is
deliberately missing for ~35% of accounts to mirror real survey
non-response.

Regenerate with `cd data && python3 generate_data.py` (writes CSVs +
`churn.db` SQLite database).

## Method

| Step | Tool | File |
|---|---|---|
| Logo & MRR churn rate by plan/contract | SQL | `sql/01_churn_rates.sql` |
| **Leakage-aware** feature engineering — rolling 3-month login average/trend and trailing support tickets via window functions, snapshotted the month *before* churn (not the churn month itself) | SQL | `sql/02_feature_engineering.sql` |
| Monthly churn trend | SQL | `sql/03_monthly_churn_trend.sql` |
| Missing-data handling, logistic regression (odds ratios), random forest, ROC/AUC comparison, account risk scoring | Python (`scikit-learn`) | `notebook/analysis.ipynb` |
| Churn-risk dashboard + ranked outreach list | Python (Plotly) | `dashboard/index.html` — **open in any browser** |

**Why the feature-engineering step matters more than it looks:** it would
be easy to build features from the churn month itself and get a model that
looks great but is useless in practice (you only know a customer's churn-
month behavior once they've already churned). This project explicitly
snapshots churned accounts' features as of the month *before* they
churned, so the model is genuinely predictive, not just descriptive.

## Key findings

- **Contract type is the single biggest lever**: Annual contracts churn
  2–3x less than Monthly, consistently across every plan tier (Starter
  Monthly: 50.6% vs. Starter Annual: 17.1% cumulative churn).
- **A random forest model reaches ~0.86 AUC** predicting churn from
  pre-churn signals alone — well above a naive baseline — using only
  rolling login trend, trailing support tickets, tenure, plan, contract,
  and NPS (with a missingness flag).
- **Declining login trend and rising support tickets are the strongest,
  most consistent drivers** across both the logistic regression's odds
  ratios and the random forest's feature importances — an actionable,
  explainable early-warning signal, not a black-box score.
- The model scores every currently-active account and buckets it into
  Low/Medium/High risk with associated MRR — see `dashboard/
  scored_accounts.csv` for the full ranked list.

## Recommendations delivered to "leadership"

1. Launch an annual-upgrade incentive targeted at Monthly Starter/Growth
   accounts — the highest-leverage, most changeable churn driver in the
   data.
2. Stand up a Customer Success alert on the two leading indicators
   (2+ consecutive months of login decline, and support ticket spikes)
   rather than waiting for a monthly churn report — by the time a monthly
   report shows the problem, the account is often already gone.
3. Prioritize outreach using the High-risk tier account list
   (`scored_accounts.csv`), sorted by MRR at risk, so limited CS bandwidth
   goes to the accounts where saving them matters most financially.

## How to talk about this in an interview

- "I deliberately avoided the most common churn-modeling mistake — using
  the churn month's own data to predict churn in that month. I engineered
  features as of the month *before* churn, so the model is actually
  usable for early intervention, not just a retrospective explanation."
- "I built both an interpretable model (logistic regression, for odds-
  ratio talking points a non-technical stakeholder can act on) and a
  higher-performing black-box model (random forest), and I can explain the
  tradeoff between them rather than defaulting to whichever scored
  higher."
- "The deliverable isn't a chart — it's a ranked, MRR-weighted account
  list a Customer Success team could act on the same day."
