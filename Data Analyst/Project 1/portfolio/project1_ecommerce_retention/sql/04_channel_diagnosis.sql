/* ============================================================
   Diagnostic query: month-over-month order volume, cancellation
   rate, avg review score, and repeat-purchase rate by acquisition
   channel. This is the query that surfaces the Aug-Sep 2025
   fulfillment problem referenced in the case study.
   ============================================================ */

WITH monthly AS (
    SELECT
        strftime('%Y-%m', o.order_date)          AS order_month,
        o.acquisition_channel_at_order            AS channel,
        COUNT(*)                                   AS total_orders,
        SUM(CASE WHEN o.order_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
        SUM(CASE WHEN o.order_status = 'Returned'  THEN 1 ELSE 0 END) AS returned_orders,
        ROUND(AVG(o.order_value), 2)               AS avg_order_value
    FROM orders o
    GROUP BY order_month, channel
),
review_join AS (
    SELECT
        strftime('%Y-%m', o.order_date)  AS order_month,
        o.acquisition_channel_at_order    AS channel,
        ROUND(AVG(r.review_score), 2)      AS avg_review_score
    FROM orders o
    JOIN reviews r ON r.order_id = o.order_id
    GROUP BY order_month, channel
)
SELECT
    m.order_month,
    m.channel,
    m.total_orders,
    m.cancelled_orders,
    ROUND(100.0 * m.cancelled_orders / m.total_orders, 1) AS cancel_rate_pct,
    ROUND(100.0 * m.returned_orders  / m.total_orders, 1) AS return_rate_pct,
    m.avg_order_value,
    rj.avg_review_score
FROM monthly m
LEFT JOIN review_join rj ON rj.order_month = m.order_month AND rj.channel = m.channel
ORDER BY m.order_month, m.channel;
