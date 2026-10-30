/* ============================================================
   Logo churn rate & MRR churn rate by plan tier and contract type.
   Logo churn = % of accounts that churned (count-based).
   MRR churn  = % of monthly recurring revenue lost to churned accounts
                (revenue-weighted -- often tells a different story than
                logo churn, since larger accounts may churn less often).
   ============================================================ */

SELECT
    plan_tier,
    contract_type,
    COUNT(*)                                             AS total_accounts,
    SUM(churned)                                           AS churned_accounts,
    ROUND(100.0 * SUM(churned) / COUNT(*), 1)              AS logo_churn_rate_pct,
    ROUND(SUM(initial_mrr), 2)                              AS total_mrr,
    ROUND(SUM(CASE WHEN churned = 1 THEN initial_mrr ELSE 0 END), 2) AS churned_mrr,
    ROUND(100.0 * SUM(CASE WHEN churned = 1 THEN initial_mrr ELSE 0 END)
                 / NULLIF(SUM(initial_mrr), 0), 1)          AS mrr_churn_rate_pct
FROM customers
GROUP BY plan_tier, contract_type
ORDER BY logo_churn_rate_pct DESC;
