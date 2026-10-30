/* ============================================================
   Builds the account-level modeling snapshot, leakage-free for BOTH
   labels: for a churned account, the snapshot is the month BEFORE
   churn; for an account flagged as an expansion candidate, the
   snapshot is the month BEFORE the expansion signal; for every other
   account, it's their most recent month (current state). This means
   both the churn model and the expansion model are trained on
   leading indicators, not the event month's own data.

   Rolling features (3-month trailing average utilization, utilization
   trend %, trailing support ticket count) computed via window
   functions over each account's ordered monthly usage history.
   ============================================================ */

WITH ordered AS (
    SELECT
        u.*,
        ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY month)                          AS month_seq,
        AVG(utilization_pct) OVER (PARTITION BY account_id ORDER BY month
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                          AS util_3mo_avg,
        LAG(utilization_pct, 3) OVER (PARTITION BY account_id ORDER BY month)                AS util_3mo_ago,
        SUM(support_tickets) OVER (PARTITION BY account_id ORDER BY month
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                          AS tickets_trailing_3mo,
        SUM(ticket_negative_sentiment) OVER (PARTITION BY account_id ORDER BY month
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                          AS negative_tickets_trailing_3mo
    FROM usage_monthly u
),
trend AS (
    SELECT *,
        CASE WHEN util_3mo_ago IS NOT NULL AND util_3mo_ago > 0
             THEN ROUND(100.0*(utilization_pct - util_3mo_ago)/util_3mo_ago, 1)
             ELSE NULL END AS util_trend_pct_3mo
    FROM ordered
),
snapshot_month AS (
    SELECT
        a.account_id,
        CASE
            WHEN a.churned = 1  THEN strftime('%Y-%m', date(a.churn_month || '-01', '-1 month'))
            WHEN a.expanded = 1 THEN strftime('%Y-%m', date(a.expansion_month || '-01', '-1 month'))
            ELSE (SELECT MAX(month) FROM usage_monthly um WHERE um.account_id = a.account_id)
        END AS snapshot_month
    FROM accounts a
)
SELECT
    a.account_id, a.plan_tier, a.contract_type, a.industry, a.region, a.seat_count, a.mrr,
    a.nps_score, a.churned, a.expanded, a.days_to_renewal,
    t.month_seq AS tenure_months_at_snapshot,
    t.utilization_pct, t.util_3mo_avg, t.util_trend_pct_3mo,
    t.tickets_trailing_3mo, t.negative_tickets_trailing_3mo
FROM accounts a
JOIN snapshot_month sm ON sm.account_id = a.account_id
JOIN trend t            ON t.account_id = a.account_id AND t.month = sm.snapshot_month;
