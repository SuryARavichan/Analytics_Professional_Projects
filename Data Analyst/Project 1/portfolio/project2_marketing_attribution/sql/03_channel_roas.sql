/* ============================================================
   Channel ROAS (Return on Ad Spend) under the linear attribution
   model, joined against actual monthly spend. This is the query
   that shows *why* the choice of attribution model matters
   operationally -- a channel can look unprofitable under
   last-touch and clearly profitable under linear/position-based.
   ============================================================ */

WITH converting_journeys AS (
    SELECT * FROM touchpoints WHERE converted = 1
),
journey_bounds AS (
    SELECT user_id, MIN(touchpoint_order) AS first_order, MAX(touchpoint_order) AS last_order, COUNT(*) AS n_touches
    FROM converting_journeys GROUP BY user_id
),
linear_credit AS (
    SELECT
        cj.channel,
        SUM(cj.conversion_value * 1.0 / jb.n_touches) AS attributed_revenue
    FROM converting_journeys cj
    JOIN journey_bounds jb ON jb.user_id = cj.user_id
    GROUP BY cj.channel
),
total_spend AS (
    SELECT channel, ROUND(SUM(spend), 2) AS total_spend, SUM(clicks) AS total_clicks
    FROM channel_spend
    GROUP BY channel
)
SELECT
    ts.channel,
    ts.total_spend,
    ROUND(COALESCE(lc.attributed_revenue, 0), 2) AS linear_attributed_revenue,
    ROUND(COALESCE(lc.attributed_revenue, 0) / NULLIF(ts.total_spend, 0), 2) AS roas
FROM total_spend ts
LEFT JOIN linear_credit lc ON lc.channel = ts.channel
ORDER BY roas DESC;
