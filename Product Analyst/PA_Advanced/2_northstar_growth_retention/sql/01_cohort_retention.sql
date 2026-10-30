/* ============================================================
   Cohort retention: for each signup-month cohort, the % of accounts
   still active at week 0/1/4/8/12/16 since signup -- restricted, per
   milestone week, to accounts whose observed_horizon reached at least
   that week (otherwise a young cohort would show fake "retention drop"
   simply because it hasn't been alive long enough to observe yet).

   weekly_activity is a long-format account x week_since_signup x active
   log; a window function (MAX(week_since_signup) per account) recovers
   each account's observed horizon so the denominator can be restricted
   correctly per milestone week.
   ============================================================ */
WITH account_horizon AS (
    SELECT account_id, MAX(week_since_signup) AS observed_horizon
    FROM weekly_activity
    GROUP BY account_id
),
cohort AS (
    SELECT a.account_id, a.signup_month, h.observed_horizon
    FROM accounts a
    JOIN account_horizon h ON h.account_id = a.account_id
),
milestones AS (
    SELECT 0 AS milestone_week UNION SELECT 1 UNION SELECT 4 UNION SELECT 8 UNION SELECT 12 UNION SELECT 16
),
eligible AS (
    SELECT c.signup_month, m.milestone_week, c.account_id
    FROM cohort c
    JOIN milestones m ON c.observed_horizon >= m.milestone_week
),
retained AS (
    SELECT e.signup_month, e.milestone_week, e.account_id,
           w.active AS was_active
    FROM eligible e
    JOIN weekly_activity w ON w.account_id = e.account_id AND w.week_since_signup = e.milestone_week
)
SELECT
    signup_month,
    milestone_week,
    COUNT(*) AS eligible_accounts,
    SUM(was_active) AS retained_accounts,
    ROUND(100.0 * SUM(was_active) / COUNT(*), 1) AS retention_pct
FROM retained
GROUP BY signup_month, milestone_week
ORDER BY signup_month, milestone_week;
