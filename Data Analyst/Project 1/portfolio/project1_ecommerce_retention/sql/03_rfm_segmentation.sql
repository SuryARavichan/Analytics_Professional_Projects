/* ============================================================
   RFM (Recency, Frequency, Monetary) customer segmentation.
   Scores each customer 1-5 on each dimension using NTILE (window
   function), then maps the 3-digit RFM score to a business-readable
   segment name. This is the query the RFM treemap in the dashboard
   is built from, and the table Python reads in for CLV modeling.
   ============================================================ */

WITH order_facts AS (
    SELECT
        customer_id,
        MAX(order_date)                 AS last_order_date,
        COUNT(DISTINCT order_id)         AS frequency,
        SUM(order_value)                 AS monetary
    FROM orders
    WHERE order_status <> 'Cancelled'
    GROUP BY customer_id
),
scored AS (
    SELECT
        customer_id,
        last_order_date,
        frequency,
        monetary,
        -- "today" fixed to the last date in the dataset for reproducibility
        CAST(julianday((SELECT MAX(order_date) FROM orders)) - julianday(last_order_date) AS INT) AS recency_days,
        NTILE(5) OVER (ORDER BY julianday((SELECT MAX(order_date) FROM orders)) - julianday(last_order_date) DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC)  AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC)   AS m_score
    FROM order_facts
)
SELECT
    customer_id,
    recency_days,
    frequency,
    ROUND(monetary, 2) AS monetary,
    r_score, f_score, m_score,
    (r_score + f_score + m_score) AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 4 AND f_score >= 3                  THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'New / Promising'
        WHEN r_score BETWEEN 2 AND 3 AND f_score >= 3        THEN 'At Risk'
        WHEN r_score <= 2 AND f_score >= 4                  THEN 'Cant Lose Them'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2  THEN 'Hibernating'
        ELSE 'Needs Attention'
    END AS rfm_segment
FROM scored
ORDER BY rfm_total DESC;
