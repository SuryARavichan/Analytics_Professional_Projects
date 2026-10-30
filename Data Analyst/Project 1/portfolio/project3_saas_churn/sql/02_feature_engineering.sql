/* ============================================================
   Builds the customer-level modeling snapshot used by the churn
   model in the notebook. For CHURNED customers, the snapshot is
   taken as of the month BEFORE they churned (so the model predicts
   churn ahead of time using leading indicators, not the churn
   month's own usage -- avoiding label leakage). For ACTIVE
   customers, the snapshot is their most recent month (i.e. "current
   risk as of today").

   Rolling features (3-month trailing average logins, login trend %,
   trailing 3-month support ticket count) are computed with window
   functions over each customer's ordered monthly usage history.
   ============================================================ */

WITH ordered AS (
    SELECT
        u.*,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month)                       AS month_seq,
        AVG(logins) OVER (PARTITION BY customer_id ORDER BY month
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                       AS logins_3mo_avg,
        LAG(logins, 3) OVER (PARTITION BY customer_id ORDER BY month)                      AS logins_3mo_ago,
        SUM(support_tickets) OVER (PARTITION BY customer_id ORDER BY month
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                       AS tickets_trailing_3mo
    FROM usage_monthly u
),
trend AS (
    SELECT *,
        CASE WHEN logins_3mo_ago IS NOT NULL AND logins_3mo_ago > 0
             THEN ROUND(100.0 * (logins - logins_3mo_ago) / logins_3mo_ago, 1)
             ELSE NULL END AS login_trend_pct_3mo
    FROM ordered
),
snapshot_month AS (
    SELECT
        c.customer_id,
        CASE
            WHEN c.churned = 1 THEN strftime('%Y-%m', date(c.churn_month || '-01', '-1 month'))
            ELSE (SELECT MAX(month) FROM usage_monthly)
        END AS snapshot_month
    FROM customers c
)
SELECT
    c.customer_id,
    c.plan_tier,
    c.contract_type,
    c.region,
    c.industry,
    c.nps_score,
    c.initial_mrr,
    c.churned,
    t.month_seq          AS tenure_months_at_snapshot,
    t.logins,
    t.logins_3mo_avg,
    t.login_trend_pct_3mo,
    t.tickets_trailing_3mo,
    t.features_used
FROM customers c
JOIN snapshot_month sm ON sm.customer_id = c.customer_id
JOIN trend t            ON t.customer_id = c.customer_id AND t.month = sm.snapshot_month;
