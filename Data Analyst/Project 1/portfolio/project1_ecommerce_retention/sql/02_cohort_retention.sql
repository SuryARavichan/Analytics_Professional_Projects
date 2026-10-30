/* ============================================================
   Monthly acquisition-cohort retention table.
   Output: cohort_month, months_since_signup, active_customers,
           cohort_size, retention_rate
   This is the query that powers the cohort heatmap in the notebook
   and dashboard. Uses window functions + CTEs (no procedural code).
   ============================================================ */

WITH cohort_map AS (
    SELECT
        customer_id,
        strftime('%Y-%m', signup_date)               AS cohort_month
    FROM customers
),
order_activity AS (
    SELECT
        o.customer_id,
        strftime('%Y-%m', o.order_date)               AS order_month
    FROM orders o
    WHERE o.order_status <> 'Cancelled'
    GROUP BY o.customer_id, order_month
),
cohort_activity AS (
    SELECT
        cm.cohort_month,
        oa.order_month,
        -- months_since_signup = integer month difference
        ( (CAST(strftime('%Y', oa.order_month || '-01') AS INT) - CAST(strftime('%Y', cm.cohort_month || '-01') AS INT)) * 12
          + (CAST(strftime('%m', oa.order_month || '-01') AS INT) - CAST(strftime('%m', cm.cohort_month || '-01') AS INT)) )
                                                        AS months_since_signup,
        cm.customer_id
    FROM cohort_map cm
    JOIN order_activity oa ON oa.customer_id = cm.customer_id
),
cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_size
    FROM cohort_map
    GROUP BY cohort_month
),
cohort_counts AS (
    SELECT
        cohort_month,
        months_since_signup,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM cohort_activity
    WHERE months_since_signup >= 0
    GROUP BY cohort_month, months_since_signup
)
SELECT
    cc.cohort_month,
    cc.months_since_signup,
    cc.active_customers,
    cs.cohort_size,
    ROUND(100.0 * cc.active_customers / cs.cohort_size, 1) AS retention_rate_pct
FROM cohort_counts cc
JOIN cohort_size cs ON cs.cohort_month = cc.cohort_month
ORDER BY cc.cohort_month, cc.months_since_signup;
