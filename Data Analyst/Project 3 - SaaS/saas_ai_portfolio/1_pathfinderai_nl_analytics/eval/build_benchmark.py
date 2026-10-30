"""
20-question PLG/SaaS analytics benchmark for PathfinderAI, mirroring
QueryMind's evaluation design: each question has a hand-verified GOLD SQL
query and a PREDICTED query representing the cached-demo LLM output. 16 of
20 are correct; 4 contain realistic text-to-SQL failure modes specific to
this schema (a fan-out/missing-DISTINCT error, a stale-data filter error,
an ambiguous SaaS-terminology resolution, and a grain/denominator error).
"""
import json

BENCH = [
    dict(id="q01", difficulty="easy",
         question="How many accounts signed up in Q3 2025 (Jul-Sep)?",
         gold="SELECT COUNT(*) AS n_accounts FROM accounts WHERE signup_date BETWEEN '2025-07-01' AND '2025-09-30';",
         predicted=None, note=None),

    dict(id="q02", difficulty="easy",
         question="What is the overall trial activation rate?",
         gold="""SELECT ROUND(100.0 * (SELECT COUNT(DISTINCT account_id) FROM product_events WHERE event_type='activated')
                 / (SELECT COUNT(*) FROM accounts), 2) AS activation_rate_pct;""",
         predicted=None, note=None),

    dict(id="q03", difficulty="medium",
         question="What is the trial-to-paid conversion rate by acquisition channel?",
         gold="""SELECT a.acquisition_channel, ROUND(100.0*SUM(CASE WHEN s.mrr>0 THEN 1 ELSE 0 END)/COUNT(*),2) AS conversion_rate_pct
                 FROM accounts a JOIN subscriptions s ON s.account_id=a.account_id
                 GROUP BY a.acquisition_channel ORDER BY conversion_rate_pct DESC;""",
         predicted=None, note=None),

    dict(id="q04", difficulty="easy",
         question="What is the current total MRR (Monthly Recurring Revenue)?",
         gold="SELECT ROUND(SUM(mrr),2) AS total_mrr FROM subscriptions WHERE status='Active';",
         predicted=None, note=None),

    dict(id="q05", difficulty="medium",
         question="Which plan tier has the highest churn rate?",
         gold="""SELECT plan_tier, ROUND(100.0*SUM(CASE WHEN status='Churned' THEN 1 ELSE 0 END)/COUNT(*),2) AS churn_rate_pct
                 FROM subscriptions WHERE plan_tier IS NOT NULL GROUP BY plan_tier ORDER BY churn_rate_pct DESC;""",
         predicted=None, note=None),

    dict(id="q06", difficulty="medium",
         question="What are the top 5 most-adopted features by number of unique accounts using them?",
         gold="""SELECT feature_name, COUNT(DISTINCT account_id) AS accounts_using
                 FROM product_events WHERE event_type='feature_used'
                 GROUP BY feature_name ORDER BY accounts_using DESC LIMIT 5;""",
         predicted=None, note=None),

    dict(id="q07", difficulty="hard",
         question="How many unique accounts used the 'automation_rules' feature?",
         gold="""SELECT COUNT(DISTINCT account_id) AS n_accounts FROM product_events
                 WHERE event_type='feature_used' AND feature_name='automation_rules';""",
         predicted="""SELECT COUNT(account_id) AS n_accounts FROM product_events
                 WHERE event_type='feature_used' AND feature_name='automation_rules';""",
         note="Missing DISTINCT: an account that used the feature many times (the data logs every usage event) "
              "is counted once per event instead of once per account, inflating the 'unique accounts' answer."),

    dict(id="q08", difficulty="medium",
         question="What is the activation rate for accounts acquired via Referral?",
         gold="""SELECT ROUND(100.0 *
                   (SELECT COUNT(DISTINCT pe.account_id) FROM product_events pe JOIN accounts a ON a.account_id=pe.account_id
                    WHERE pe.event_type='activated' AND a.acquisition_channel='Referral')
                   / (SELECT COUNT(*) FROM accounts WHERE acquisition_channel='Referral'), 2) AS activation_rate_pct;""",
         predicted=None, note=None),

    dict(id="q09", difficulty="easy",
         question="What is the average MRR by plan tier for active accounts?",
         gold="SELECT plan_tier, ROUND(AVG(mrr),2) AS avg_mrr FROM subscriptions WHERE status='Active' GROUP BY plan_tier;",
         predicted=None, note=None),

    dict(id="q10", difficulty="easy",
         question="What percentage of paid accounts are on annual billing?",
         gold="SELECT ROUND(100.0*SUM(CASE WHEN billing_cycle='Annual' THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_annual FROM subscriptions WHERE mrr>0;",
         predicted=None, note=None),

    dict(id="q11", difficulty="hard",
         question="What is the total current MRR by plan tier?",
         gold="SELECT plan_tier, ROUND(SUM(mrr),2) AS total_mrr FROM subscriptions WHERE status='Active' GROUP BY plan_tier;",
         predicted="SELECT plan_tier, ROUND(SUM(mrr),2) AS total_mrr FROM subscriptions WHERE mrr > 0 GROUP BY plan_tier;",
         note="Stale-data filter error: this includes Churned accounts (WHERE mrr>0 doesn't exclude them -- their mrr "
              "column still holds the value from when they were paying), overstating CURRENT MRR by including "
              "revenue from accounts that already cancelled."),

    dict(id="q12", difficulty="easy",
         question="How many accounts are in the '1000+' employee size bracket?",
         gold="SELECT COUNT(*) AS n FROM accounts WHERE employee_size_bracket='1000+';",
         predicted=None, note=None),

    dict(id="q13", difficulty="hard",
         question="How many Enterprise accounts do we have?",
         gold="SELECT COUNT(*) AS n FROM subscriptions WHERE plan_tier='Enterprise';",
         predicted="SELECT COUNT(*) AS n FROM accounts WHERE employee_size_bracket='1000+';",
         note="Ambiguous SaaS terminology resolved the wrong way: 'Enterprise accounts' in a subscription-business "
              "context almost always means accounts on the Enterprise PRICING PLAN, but the model resolved it to "
              "company employee-size instead -- a different, silently plausible interpretation that materially "
              "changes the answer (these are not the same set of accounts)."),

    dict(id="q14", difficulty="hard",
         question="What is the average time from signup to activation, in days?",
         gold="""WITH act AS (SELECT account_id, MIN(event_date) AS act_date FROM product_events
                 WHERE event_type='activated' GROUP BY account_id)
                 SELECT ROUND(AVG(julianday(act.act_date) - julianday(a.signup_date)),2) AS avg_days_to_activation
                 FROM act JOIN accounts a ON a.account_id=act.account_id;""",
         predicted=None, note=None),

    dict(id="q15", difficulty="medium",
         question="Which industry has the highest paid conversion rate?",
         gold="""SELECT a.industry, ROUND(100.0*SUM(CASE WHEN s.mrr>0 THEN 1 ELSE 0 END)/COUNT(*),2) AS conversion_rate_pct
                 FROM accounts a JOIN subscriptions s ON s.account_id=a.account_id
                 GROUP BY a.industry ORDER BY conversion_rate_pct DESC LIMIT 1;""",
         predicted=None, note=None),

    dict(id="q16", difficulty="hard",
         question="What percentage of activated accounts convert to paid?",
         gold="""WITH activated AS (SELECT DISTINCT account_id FROM product_events WHERE event_type='activated')
                 SELECT ROUND(100.0*SUM(CASE WHEN s.mrr>0 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct
                 FROM activated act JOIN subscriptions s ON s.account_id=act.account_id;""",
         predicted="""SELECT ROUND(100.0*SUM(CASE WHEN s.mrr>0 THEN 1 ELSE 0 END)/COUNT(*),2) AS pct
                 FROM product_events pe JOIN subscriptions s ON s.account_id=pe.account_id;""",
         note="Grain/denominator error: joins the raw product_events table (many rows per account -- every feature "
              "use, not just the one 'activated' row) instead of a deduplicated activated-accounts set, so the "
              "denominator massively overcounts and the percentage is systematically wrong."),

    dict(id="q17", difficulty="easy",
         question="What is the total number of feature-usage events per month?",
         gold="""SELECT strftime('%Y-%m', event_date) AS month, COUNT(*) AS n_events
                 FROM product_events WHERE event_type='feature_used' GROUP BY month ORDER BY month;""",
         predicted=None, note=None),

    dict(id="q18", difficulty="medium",
         question="Which acquisition channel produces the highest average MRR per converted account?",
         gold="""SELECT a.acquisition_channel, ROUND(AVG(s.mrr),2) AS avg_mrr
                 FROM accounts a JOIN subscriptions s ON s.account_id=a.account_id
                 WHERE s.mrr>0 GROUP BY a.acquisition_channel ORDER BY avg_mrr DESC LIMIT 1;""",
         predicted=None, note=None),

    dict(id="q19", difficulty="medium",
         question="How many accounts churned within 90 days of converting to paid?",
         gold="""SELECT COUNT(*) AS n FROM subscriptions
                 WHERE status='Churned' AND julianday(cancel_date) - julianday(start_date) <= 90;""",
         predicted=None, note=None),

    dict(id="q20", difficulty="hard",
         question="What is the feature adoption rate (as a % of activated accounts) for 'api_access'?",
         gold="""WITH activated AS (SELECT DISTINCT account_id FROM product_events WHERE event_type='activated')
                 SELECT ROUND(100.0*COUNT(DISTINCT pe.account_id) / (SELECT COUNT(*) FROM activated), 2) AS adoption_rate_pct
                 FROM product_events pe JOIN activated act ON act.account_id=pe.account_id
                 WHERE pe.event_type='feature_used' AND pe.feature_name='api_access';""",
         predicted=None, note=None),
]

for row in BENCH:
    if row["predicted"] is None:
        row["predicted"] = row["gold"]

with open("benchmark.json", "w") as f:
    json.dump(BENCH, f, indent=2)

cached_sql = {row["id"]: row["predicted"] for row in BENCH}
with open("cached_sql_responses.json", "w") as f:
    json.dump(cached_sql, f, indent=2)

print(f"Wrote benchmark.json ({len(BENCH)} questions)")
print(f"Injected failures: {[r['id'] for r in BENCH if r['note']]}")
