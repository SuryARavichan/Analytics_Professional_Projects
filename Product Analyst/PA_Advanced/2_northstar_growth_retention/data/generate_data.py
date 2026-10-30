"""
Generates a 12-month Flowdesk signup cohort (4,000 accounts) with an
internally consistent generative model: every account has a single latent
"engagement propensity" that drives its activation-funnel behavior, its
week-1/week-2 usage metrics (the North Star metric candidates), its
weekly churn hazard (for survival analysis), and its current-state
recency/frequency/breadth behavior (for engagement segmentation) -- so
every downstream analysis is measuring genuinely correlated signal, not
independently-drawn noise dressed up to look related.

Produces:
  accounts.csv        — one row per account: signup week, onboarding
                         path, channel, plan tier, funnel-step flags,
                         North Star candidate metrics, survival duration/
                         censoring, and current-state RFM features.
  weekly_activity.csv — long-format account x week-since-signup activity
                         log (derived from each account's survival
                         duration), for a genuine SQL cohort-retention
                         query with window functions.

The built-in story: "Guided Setup" onboarding (30% of signups) shifts
engagement propensity up, which improves activation, retention, AND
survival -- a single, coherent causal thread through the whole dataset,
discoverable independently through the funnel, the retention curve, AND
the survival analysis.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(303)

N = 4000
ANCHOR_WEEK = 52          # "today" -- the most recent completed week
MAX_OBS_HORIZON = 26      # cap observation horizon so the retention curve has a consistent max range

ONBOARDING_PATHS = ["Guided Setup", "Self-Serve"]
ONBOARDING_SHARE = [0.30, 0.70]
CHANNELS = ["Organic", "Paid", "Referral", "Sales-Assisted"]
CHANNEL_SHARE = [0.35, 0.30, 0.20, 0.15]
PLAN_TIERS = ["Starter", "Growth", "Enterprise"]
PLAN_SHARE = [0.50, 0.35, 0.15]

signup_week = rng.integers(1, ANCHOR_WEEK + 1, N)
onboarding_path = rng.choice(ONBOARDING_PATHS, N, p=ONBOARDING_SHARE)
channel = rng.choice(CHANNELS, N, p=CHANNEL_SHARE)
plan_tier = rng.choice(PLAN_TIERS, N, p=PLAN_SHARE)

# ---- the one latent variable that drives everything downstream ----
propensity = rng.normal(0, 1, N)
propensity += np.where(onboarding_path == "Guided Setup", 0.65, 0.0)
propensity += np.select(
    [channel == "Sales-Assisted", channel == "Referral", channel == "Paid"],
    [0.35, 0.15, -0.20], default=0.0,
)

# ---- survival: weekly churn hazard, lower for higher propensity ----
base_hazard = 0.085
hazard = np.clip(base_hazard * np.exp(-0.55 * propensity), 0.01, 0.35)
true_time_to_churn = rng.geometric(hazard)  # weeks survived before churning (>=1)

observed_horizon = np.minimum(ANCHOR_WEEK - signup_week, MAX_OBS_HORIZON)
duration = np.minimum(true_time_to_churn, observed_horizon)
event_observed = (true_time_to_churn <= observed_horizon).astype(int)  # 1=churn observed, 0=censored (still active)

# ---- activation funnel (week 1) ----
p_first_login = 1 / (1 + np.exp(-(0.9 + 0.5 * propensity + np.where(channel == "Sales-Assisted", 0.6, 0.0))))
first_login = rng.binomial(1, p_first_login)

lambda1 = np.exp(0.9 + 0.5 * propensity) * first_login
week1_core_actions = rng.poisson(np.clip(lambda1, 0, None))
core_action_completed = (week1_core_actions >= 1).astype(int)
activated = ((first_login == 1) & (week1_core_actions >= 3)).astype(int)

p_convert = 1 / (1 + np.exp(-(-1.4 + 2.1 * activated + 0.4 * propensity)))
converted_to_paid = rng.binomial(1, p_convert)

# ---- North Star candidate metrics (some genuinely predictive, one deliberately weak) ----
week2_core_actions = week1_core_actions + rng.poisson(np.clip(lambda1 * 0.8, 0, None))
workspaces_created_wk1 = rng.poisson(np.clip(0.35 * np.exp(0.15 * propensity), 0, None))  # weak/noisy signal on purpose

# ---- current-state RFM features, for engagement segmentation (currently-active accounts only) ----
recent_4wk_actions = rng.poisson(np.clip(np.exp(1.2 + 0.6 * propensity), 0, None))
features_used_recent = 1 + rng.binomial(7, p=np.clip(1 / (1 + np.exp(-(-0.5 + 0.4 * propensity))), 0.02, 0.98))
recency_weeks_since_active = rng.poisson(np.clip(1.5 * np.exp(-0.5 * propensity), 0, None)).clip(0, 6)

accounts = pd.DataFrame({
    "account_id": [f"A{200000+i}" for i in range(N)],
    "signup_week": signup_week, "signup_month": ((signup_week - 1) // 4) + 1,
    "onboarding_path": onboarding_path, "channel": channel, "plan_tier": plan_tier,
    "first_login": first_login, "week1_core_actions": week1_core_actions,
    "core_action_completed": core_action_completed, "activated": activated,
    "converted_to_paid": converted_to_paid,
    "week2_core_actions": week2_core_actions, "workspaces_created_wk1": workspaces_created_wk1,
    "duration_weeks": duration, "event_observed": event_observed, "observed_horizon": observed_horizon,
    "recent_4wk_actions": recent_4wk_actions, "features_used_recent": features_used_recent,
    "recency_weeks_since_active": recency_weeks_since_active,
})
accounts.to_csv("accounts.csv", index=False)

# ---- weekly_activity long-format log, derived from survival duration, for SQL cohort retention ----
rows = []
for acct_id, horizon, dur in zip(accounts.account_id, accounts.observed_horizon, accounts.duration_weeks):
    for w in range(0, int(horizon) + 1):
        rows.append({"account_id": acct_id, "week_since_signup": w, "active": int(w < dur)})
weekly_activity = pd.DataFrame(rows)
weekly_activity.to_csv("weekly_activity.csv", index=False)

print(f"accounts: {len(accounts):,} | activation rate {accounts.activated.mean():.1%} | "
      f"paid conversion {accounts.converted_to_paid.mean():.1%} | "
      f"churn observed {accounts.event_observed.mean():.1%} (rest censored/still active)")
print(f"  by onboarding path -- activation: {accounts.groupby('onboarding_path').activated.mean().to_dict()}")
print(f"weekly_activity: {len(weekly_activity):,} rows")

import sqlite3
conn = sqlite3.connect("northstar.db")
accounts.to_sql("accounts", conn, if_exists="replace", index=False)
weekly_activity.to_sql("weekly_activity", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote accounts.csv, weekly_activity.csv + northstar.db")
