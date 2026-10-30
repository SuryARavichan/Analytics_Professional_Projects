/* ============================================================
   A/B test topline + segment cut (device) results.
   Aggregates to conversion rate, avg order value, and avg load
   time per variant, both overall and split by device -- this is
   the query that surfaces the heterogeneous treatment effect
   (mobile lift vs flat desktop) that a blended topline number hides.
   ============================================================ */

WITH overall AS (
    SELECT
        'ALL' AS device,
        variant,
        COUNT(*)                                   AS sessions,
        SUM(converted)                              AS conversions,
        ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_rate_pct,
        ROUND(AVG(CASE WHEN converted = 1 THEN order_value END), 2) AS avg_order_value,
        ROUND(AVG(page_load_time_ms), 0)            AS avg_load_time_ms
    FROM ab_sessions
    GROUP BY variant
),
by_device AS (
    SELECT
        device,
        variant,
        COUNT(*)                                   AS sessions,
        SUM(converted)                              AS conversions,
        ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_rate_pct,
        ROUND(AVG(CASE WHEN converted = 1 THEN order_value END), 2) AS avg_order_value,
        ROUND(AVG(page_load_time_ms), 0)            AS avg_load_time_ms
    FROM ab_sessions
    GROUP BY device, variant
)
SELECT * FROM overall
UNION ALL
SELECT * FROM by_device
ORDER BY device, variant;
