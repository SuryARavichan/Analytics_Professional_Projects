"""
Generates a synthetic product-led-growth (PLG) SaaS dataset for
"Flowdesk," a fictional team-collaboration app: 4 related tables
(accounts, users, subscriptions, product_events) with enough join
complexity and funnel structure (signup -> trial -> activation ->
paid conversion -> expansion/churn) to make natural-language-to-SQL
genuinely exercise real PLG analytics questions, not just simple filters.
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

rng = np.random.default_rng(71)

START = datetime(2024, 4, 1)
END = datetime(2025, 12, 31)

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REGION_W = [0.46, 0.28, 0.17, 0.09]
CHANNELS = ["Organic Search", "Paid Search", "Content/SEO", "Referral", "Product Hunt/Community"]
CHANNEL_W = [0.24, 0.22, 0.18, 0.20, 0.16]
# activation & conversion multipliers by channel (referral/community users engage deeper)
CHANNEL_ACTIVATION_MULT = {"Organic Search": 1.0, "Paid Search": 0.78, "Content/SEO": 1.05,
                            "Referral": 1.35, "Product Hunt/Community": 1.15}
INDUSTRIES = ["Software", "Marketing Agency", "Professional Services", "E-commerce", "Education", "Nonprofit"]
SIZE_BRACKETS = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
SIZE_W = [0.38, 0.32, 0.18, 0.09, 0.03]
PLAN_TIERS = ["Starter", "Growth", "Enterprise"]
PLAN_MRR = {"Starter": (12, 49), "Growth": (99, 399), "Enterprise": (500, 2500)}

N_ACCOUNTS = 3200
account_ids = [f"ACC{40000+i}" for i in range(N_ACCOUNTS)]
signup_offsets = rng.integers(0, (END - START).days - 30, N_ACCOUNTS)
signup_dates = [START + timedelta(days=int(o)) for o in signup_offsets]
channel = rng.choice(CHANNELS, size=N_ACCOUNTS, p=CHANNEL_W)
region = rng.choice(REGIONS, size=N_ACCOUNTS, p=REGION_W)
industry = rng.choice(INDUSTRIES, size=N_ACCOUNTS)
size_bracket = rng.choice(SIZE_BRACKETS, size=N_ACCOUNTS, p=SIZE_W)

accounts = pd.DataFrame({
    "account_id": account_ids, "company_name": [f"Company {i}" for i in range(N_ACCOUNTS)],
    "signup_date": [d.date() for d in signup_dates], "acquisition_channel": channel,
    "region": region, "industry": industry, "employee_size_bracket": size_bracket,
})

# ---------------------------------------------------------------------
# USERS (1-40 per account depending on size bracket)
# ---------------------------------------------------------------------
SIZE_USER_RANGE = {"1-10": (1,4), "11-50": (2,10), "51-200": (4,25), "201-1000": (8,45), "1000+": (15,80)}
user_rows, user_ctr = [], 1
for _, acc in accounts.iterrows():
    lo, hi = SIZE_USER_RANGE[acc["employee_size_bracket"]]
    n_users = int(rng.integers(lo, hi+1))
    signup_dt = pd.Timestamp(acc["signup_date"])
    for u in range(n_users):
        join_offset = 0 if u == 0 else int(rng.integers(0, 60))
        user_rows.append({
            "user_id": f"U{700000+user_ctr}", "account_id": acc["account_id"],
            "is_admin": int(u == 0), "user_signup_date": (signup_dt + timedelta(days=join_offset)).date(),
        })
        user_ctr += 1
users = pd.DataFrame(user_rows)

# ---------------------------------------------------------------------
# FUNNEL: trial -> activation -> paid conversion -> plan/expansion/churn
# ---------------------------------------------------------------------
sub_rows, event_rows = [], []
event_ctr = 1

for _, acc in accounts.iterrows():
    signup_dt = pd.Timestamp(acc["signup_date"])
    mult = CHANNEL_ACTIVATION_MULT[acc["acquisition_channel"]]
    acc_users = users[users.account_id == acc["account_id"]]

    # signup event
    event_rows.append({"event_id": f"EV{event_ctr}", "account_id": acc["account_id"],
                        "user_id": acc_users.iloc[0].user_id, "event_date": signup_dt.date(),
                        "event_type": "signup", "feature_name": None}); event_ctr += 1

    # activation: completes onboarding + creates first project within 7 days
    activation_prob = np.clip(0.52 * mult + rng.normal(0, 0.05), 0.05, 0.95)
    activated = rng.random() < activation_prob
    if activated:
        act_date = signup_dt + timedelta(days=int(rng.integers(0, 7)))
        if act_date <= pd.Timestamp(END):
            event_rows.append({"event_id": f"EV{event_ctr}", "account_id": acc["account_id"],
                                "user_id": acc_users.iloc[0].user_id, "event_date": act_date.date(),
                                "event_type": "activated", "feature_name": "onboarding_complete"}); event_ctr += 1

    # ongoing feature usage events for activated accounts over their lifetime
    FEATURES = ["task_board", "automation_rules", "integrations", "reporting_dashboard",
                "team_chat", "time_tracking", "api_access", "custom_fields"]
    FEATURE_ADOPTION_BASE = {"task_board": 0.85, "team_chat": 0.62, "reporting_dashboard": 0.44,
                              "automation_rules": 0.30, "integrations": 0.34, "time_tracking": 0.28,
                              "custom_fields": 0.22, "api_access": 0.12}
    lifetime_days = (min(pd.Timestamp(END), signup_dt + timedelta(days=int(rng.integers(60, 620)))) - signup_dt).days
    if activated:
        for feat, base_p in FEATURE_ADOPTION_BASE.items():
            p = np.clip(base_p * mult, 0.02, 0.97)
            if rng.random() < p:
                n_uses = int(rng.poisson(lam=4)) + 1
                for _ in range(min(n_uses, 12)):
                    ev_date = signup_dt + timedelta(days=int(rng.integers(1, max(2, lifetime_days))))
                    if ev_date <= pd.Timestamp(END):
                        event_rows.append({"event_id": f"EV{event_ctr}", "account_id": acc["account_id"],
                                            "user_id": rng.choice(acc_users.user_id.values),
                                            "event_date": ev_date.date(), "event_type": "feature_used",
                                            "feature_name": feat}); event_ctr += 1

    # ---- conversion to paid ----
    conversion_prob = np.clip((0.28 if activated else 0.04) * mult + rng.normal(0, 0.04), 0.01, 0.85)
    converted = rng.random() < conversion_prob
    status, plan, mrr, start_date, cancel_date, billing = "Trial/Not Converted", None, 0.0, None, None, None

    if converted:
        conv_date = signup_dt + timedelta(days=int(rng.integers(3, 30)))
        if conv_date <= pd.Timestamp(END):
            plan = rng.choice(PLAN_TIERS, p=[0.55, 0.34, 0.11] if acc["employee_size_bracket"] in ("1-10","11-50")
                               else [0.20, 0.48, 0.32])
            mrr = round(rng.uniform(*PLAN_MRR[plan]), 2)
            billing = rng.choice(["Monthly", "Annual"], p=[0.62, 0.38])
            start_date = conv_date.date()
            event_rows.append({"event_id": f"EV{event_ctr}", "account_id": acc["account_id"],
                                "user_id": acc_users.iloc[0].user_id, "event_date": conv_date.date(),
                                "event_type": "converted_to_paid", "feature_name": plan}); event_ctr += 1

            # churn: paid accounts may cancel later, higher chance for Starter/Monthly, lower for Enterprise/Annual
            churn_base = {"Starter": 0.34, "Growth": 0.17, "Enterprise": 0.07}[plan]
            churn_mult = 0.55 if billing == "Annual" else 1.3
            if rng.random() < churn_base * churn_mult:
                days_to_churn = int(rng.integers(45, 500))
                churn_dt = conv_date + timedelta(days=days_to_churn)
                if churn_dt <= pd.Timestamp(END):
                    cancel_date = churn_dt.date()
                    status = "Churned"
                    event_rows.append({"event_id": f"EV{event_ctr}", "account_id": acc["account_id"],
                                        "user_id": acc_users.iloc[0].user_id, "event_date": churn_dt.date(),
                                        "event_type": "cancelled", "feature_name": plan}); event_ctr += 1
                else:
                    status = "Active"
            else:
                status = "Active"
        else:
            converted = False

    sub_rows.append({
        "account_id": acc["account_id"], "status": status, "plan_tier": plan, "mrr": mrr,
        "billing_cycle": billing, "start_date": start_date, "cancel_date": cancel_date,
    })

subscriptions = pd.DataFrame(sub_rows)
product_events = pd.DataFrame(event_rows)

print(f"accounts={len(accounts)}  users={len(users)}  subscriptions={len(subscriptions)}  "
      f"events={len(product_events)}  paid_accounts={(subscriptions.mrr>0).sum()}")

for df, name in [(accounts,"accounts"), (users,"users"), (subscriptions,"subscriptions"), (product_events,"product_events")]:
    df.to_csv(f"{name}.csv", index=False)

conn = sqlite3.connect("flowdesk.db")
accounts.to_sql("accounts", conn, if_exists="replace", index=False)
users.to_sql("users", conn, if_exists="replace", index=False)
subscriptions.to_sql("subscriptions", conn, if_exists="replace", index=False)
product_events.to_sql("product_events", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote CSVs + flowdesk.db")
