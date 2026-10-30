/* ============================================================
   Activation funnel step-through, overall and by onboarding path:
   signup -> first_login -> core_action_completed -> activated ->
   converted_to_paid, with step-over-step conversion rates so the
   biggest drop-off point is visible directly in the output.
   ============================================================ */
SELECT
    onboarding_path,
    COUNT(*) AS signups,
    SUM(first_login) AS first_login,
    ROUND(100.0 * SUM(first_login) / COUNT(*), 1) AS pct_first_login,
    SUM(core_action_completed) AS core_action_completed,
    ROUND(100.0 * SUM(core_action_completed) / NULLIF(SUM(first_login), 0), 1) AS pct_first_login_to_core_action,
    SUM(activated) AS activated,
    ROUND(100.0 * SUM(activated) / NULLIF(SUM(core_action_completed), 0), 1) AS pct_core_action_to_activated,
    SUM(converted_to_paid) AS converted_to_paid,
    ROUND(100.0 * SUM(converted_to_paid) / NULLIF(SUM(activated), 0), 1) AS pct_activated_to_converted,
    ROUND(100.0 * SUM(converted_to_paid) / COUNT(*), 1) AS pct_signup_to_converted_overall
FROM accounts
GROUP BY onboarding_path
ORDER BY pct_signup_to_converted_overall DESC;
