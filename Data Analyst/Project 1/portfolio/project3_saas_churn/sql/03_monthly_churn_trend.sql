/* ============================================================
   Monthly logo churn trend: of the accounts active at the start
   of each month, what % churned during that month? Powers the
   churn-rate-over-time line chart in the notebook/dashboard.
   ============================================================ */

WITH months AS (
    SELECT DISTINCT month FROM usage_monthly
),
active_at_start AS (
    -- an account is "active at start of month M" if it signed up before M
    -- and (never churned) or (churned in a month >= M)
    SELECT
        m.month,
        COUNT(*) AS accounts_active_at_start
    FROM months m
    JOIN customers c
      ON strftime('%Y-%m', c.signup_date) < m.month
     AND (c.churned = 0 OR c.churn_month >= m.month)
    GROUP BY m.month
),
churned_in_month AS (
    SELECT churn_month AS month, COUNT(*) AS accounts_churned
    FROM customers
    WHERE churned = 1
    GROUP BY churn_month
)
SELECT
    a.month,
    a.accounts_active_at_start,
    COALESCE(ci.accounts_churned, 0)                                    AS accounts_churned,
    ROUND(100.0 * COALESCE(ci.accounts_churned, 0) / a.accounts_active_at_start, 2) AS monthly_logo_churn_rate_pct
FROM active_at_start a
LEFT JOIN churned_in_month ci ON ci.month = a.month
ORDER BY a.month;
