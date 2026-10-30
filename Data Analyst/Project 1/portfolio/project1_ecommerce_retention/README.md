# Project 1 — Customer Retention & Cohort Analysis (DTC E-commerce)

**One-line pitch:** Diagnosed a $-relevant retention problem in a 2-year e-commerce
dataset using SQL window functions, RFM segmentation, and hypothesis testing —
then packaged the findings into an interactive stakeholder dashboard.

## Business problem

A direct-to-consumer retailer's leadership wants to know three things:

1. How well are we actually retaining customers, cohort by cohort?
2. Which acquisition channels bring in customers worth keeping — so marketing
   spend can be reallocated toward quality, not just volume?
3. Something felt "off" in Q3 2025 — did retention actually take a hit, and why?

## Data

Synthetic but realistic 2-year (Jan 2024–Dec 2025) order history for 6,000
customers / ~22,500 orders across 6 product categories and 5 acquisition
channels, modeled after a real DTC order schema (`customers`, `orders`,
`order_items`, `payments`, `reviews`). See `data/generate_data.py` for the
generation logic and the realistic behavioral rules baked into it (channel
loyalty differences, category-specific repurchase cycles, and an injected
fulfillment incident).

Regenerate anytime with:
```bash
cd data && python3 generate_data.py
```
This produces `ecommerce.db` (SQLite) plus flat CSVs, so the project can be
worked in SQL, pandas, Excel, or pointed at directly from Tableau/Power BI.

## Method

| Step | Tool | File |
|---|---|---|
| Schema + indexes | SQL | `sql/01_schema.sql` |
| Monthly cohort retention (CTEs + date math) | SQL | `sql/02_cohort_retention.sql` |
| RFM segmentation (`NTILE()` window functions) | SQL | `sql/03_rfm_segmentation.sql` |
| Channel-level incident diagnostics | SQL | `sql/04_channel_diagnosis.sql` |
| Cohort heatmap, channel retention, CLV proxy, chi-square test, incident chart | Python | `notebook/analysis.ipynb` |
| Interactive stakeholder dashboard | Python (Plotly) | `dashboard/index.html` — **open this file in any browser** |

## Key findings

- **Retention decays fast, then stabilizes** around an 8-12% "loyal core" by
  month 6, with ~30% of any cohort returning within the first month.
- **Acquisition channel significantly predicts loyalty** (chi-square,
  p < 0.001): Referral and Organic Search customers are ~2x as likely to
  become repeat buyers as Paid Social customers.
- **RFM segmentation surfaces a clear "At Risk" win-back opportunity** —
  large, previously high-value, not yet fully churned.
- **A fulfillment issue in Aug–Sep 2025 is visible in the data**:
  cancellation rate roughly doubled and average review score dropped
  ~0.8–0.9 points in that window, and it shows up directly as a dip in
  retention for cohorts acquired during that period.

## Recommendations delivered to "leadership"

1. Shift a portion of Paid Social budget toward referral incentives and
   lifecycle email — cheaper to retain loyal traffic than to keep
   re-acquiring low-loyalty traffic.
2. Launch a win-back campaign targeted at the "At Risk" RFM segment before
   they fully churn.
3. Stand up a lightweight ops monitoring view (cancel rate + review score by
   week, with an alert threshold) so a future fulfillment problem is caught
   before it erodes a full cohort's lifetime value.

## Rebuilding the dashboard in Tableau / Power BI

The Plotly HTML dashboard (`dashboard/index.html`) is the zero-install
deliverable. To rebuild the same views as a live Tableau or Power BI
dashboard for a portfolio piece that shows those tools specifically:

1. Connect Tableau/Power BI directly to `data/ecommerce.db` (or the CSVs) —
   both tools have native SQLite/CSV connectors.
2. Build a calculated field for `months_since_signup` (Tableau: `DATEDIFF`;
   Power BI: a DAX measure), or import the pre-computed
   `sql/02_cohort_retention.sql` result as a query/custom SQL source.
3. Recreate the four views: cohort heatmap (rows=cohort month,
   columns=months-since-signup, color=retention%), a horizontal bar of
   repeat-rate by channel, a bar of RFM segment sizes, and a dual-axis line
   chart of cancel-rate vs. review score with a highlighted Aug–Sep 2025
   band.

## How to talk about this in an interview

- "I built the full pipeline myself — schema design, SQL with window
  functions, statistical testing in Python, and a dashboard — so I can speak
  to every layer, not just the chart at the end."
- "I didn't just report retention was low — I used a chi-square test to show
  the channel effect was statistically significant, not noise, which
  changes how confidently you can act on it."
- "I traced a KPI dip back to an operational root cause using the data
  itself, which is the kind of diagnosis stakeholders actually need, not
  just a dashboard."
