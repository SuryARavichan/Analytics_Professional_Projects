/* ============================================================
   Topline health KPIs: churn and expansion rate by plan/contract,
   plus MRR at risk (churned) and MRR expanded, for the executive
   summary numbers.
   ============================================================ */
SELECT
    plan_tier, contract_type,
    COUNT(*) AS total_accounts,
    SUM(churned) AS churned_accounts,
    ROUND(100.0*SUM(churned)/COUNT(*),1) AS churn_rate_pct,
    SUM(expanded) AS expanded_accounts,
    ROUND(100.0*SUM(expanded)/COUNT(*),1) AS expansion_rate_pct,
    ROUND(SUM(CASE WHEN churned=1 THEN mrr ELSE 0 END),2) AS mrr_churned,
    ROUND(SUM(CASE WHEN expanded=1 THEN mrr ELSE 0 END),2) AS mrr_from_expansion_candidates
FROM accounts
GROUP BY plan_tier, contract_type
ORDER BY churn_rate_pct DESC;
