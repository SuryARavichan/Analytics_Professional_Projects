/* ============================================================
   Multi-touch attribution: credit assignment by model, per channel.
   Computes conversions credited under 4 models using pure SQL
   window functions -- no external attribution library needed:

     - First-touch   : 100% credit to touchpoint_order = 1
     - Last-touch     : 100% credit to touchpoint_order = MAX(order) in journey
     - Linear          : credit split evenly across every touchpoint
     - Position-based (U-shaped): 40% first / 40% last / 20% split across middle
   ============================================================ */

WITH converting_journeys AS (
    SELECT * FROM touchpoints WHERE converted = 1
),
journey_bounds AS (
    SELECT
        user_id,
        MIN(touchpoint_order) AS first_order,
        MAX(touchpoint_order) AS last_order,
        COUNT(*)              AS n_touches
    FROM converting_journeys
    GROUP BY user_id
),
credited AS (
    SELECT
        cj.channel,
        cj.user_id,
        cj.touchpoint_order,
        cj.conversion_value,
        jb.first_order,
        jb.last_order,
        jb.n_touches,

        -- First-touch credit
        CASE WHEN cj.touchpoint_order = jb.first_order THEN cj.conversion_value ELSE 0 END AS credit_first_touch,

        -- Last-touch credit
        CASE WHEN cj.touchpoint_order = jb.last_order THEN cj.conversion_value ELSE 0 END AS credit_last_touch,

        -- Linear credit: split evenly
        cj.conversion_value * 1.0 / jb.n_touches AS credit_linear,

        -- Position-based (U-shaped): 40% first, 40% last, 20% shared across middles
        CASE
            WHEN jb.n_touches = 1 THEN cj.conversion_value                         -- single touch gets it all
            WHEN cj.touchpoint_order = jb.first_order THEN cj.conversion_value * 0.4
            WHEN cj.touchpoint_order = jb.last_order  THEN cj.conversion_value * 0.4
            ELSE cj.conversion_value * 0.2 / NULLIF(jb.n_touches - 2, 0)
        END AS credit_position_based
    FROM converting_journeys cj
    JOIN journey_bounds jb ON jb.user_id = cj.user_id
)
SELECT
    channel,
    ROUND(SUM(credit_first_touch), 2)     AS revenue_first_touch,
    ROUND(SUM(credit_last_touch), 2)      AS revenue_last_touch,
    ROUND(SUM(credit_linear), 2)          AS revenue_linear,
    ROUND(SUM(credit_position_based), 2)  AS revenue_position_based
FROM credited
GROUP BY channel
ORDER BY revenue_linear DESC;
