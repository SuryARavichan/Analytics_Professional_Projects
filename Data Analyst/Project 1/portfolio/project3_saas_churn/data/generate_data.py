"""
Generates a realistic (synthetic) B2B SaaS subscription + usage dataset for
the Churn Prediction & Diagnosis project.

Story baked in (the signal a churn model should recover):
  - Monthly contracts churn far more than Annual contracts.
  - Starter plan churns more than Growth; Enterprise churns least.
  - A declining login trend over a customer's most recent 3 months is the
    single strongest engagement-based predictor.
  - Elevated support tickets in the last 90 days predicts churn.
  - Low NPS predicts churn, but is missing for ~35% of accounts (realistic
    survey non-response) -- a legitimate missing-data / imputation talking
    point for the notebook.

Output: customers.csv, usage_monthly.csv, and a churn.db SQLite database
with the same tables, covering 24 months (2024-01 through 2025-12).
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

rng = np.random.default_rng(11)

N_CUSTOMERS = 2600
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
ALL_MONTHS = pd.date_range(START, END, freq="MS")

PLAN_TIERS = ["Starter", "Growth", "Enterprise"]
PLAN_WEIGHTS = [0.45, 0.38, 0.17]
PLAN_MRR = {"Starter": (49, 199), "Growth": (200, 899), "Enterprise": (900, 4500)}
# base monthly churn hazard by plan (probability of churning in a given month, before other factors)
PLAN_BASE_HAZARD = {"Starter": 0.026, "Growth": 0.013, "Enterprise": 0.006}

CONTRACT_TYPES = ["Monthly", "Annual"]
CONTRACT_WEIGHTS = [0.55, 0.45]
CONTRACT_HAZARD_MULT = {"Monthly": 1.6, "Annual": 0.4}

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.48, 0.27, 0.16, 0.09]
INDUSTRIES = ["Software", "Retail/E-comm", "Financial Services", "Healthcare", "Manufacturing", "Media"]

# ------------------------------------------------------------------
# 1. CUSTOMERS
# ------------------------------------------------------------------
customer_ids = [f"ACCT{9000+i}" for i in range(N_CUSTOMERS)]
plan_tier = rng.choice(PLAN_TIERS, size=N_CUSTOMERS, p=PLAN_WEIGHTS)
contract_type = rng.choice(CONTRACT_TYPES, size=N_CUSTOMERS, p=CONTRACT_WEIGHTS)
region = rng.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)
industry = rng.choice(INDUSTRIES, size=N_CUSTOMERS)

# signup month spread across the first 20 months (so every customer has usage history)
signup_month_idx = rng.integers(0, 20, N_CUSTOMERS)
signup_dates = [ALL_MONTHS[i] for i in signup_month_idx]

initial_mrr = np.array([round(rng.uniform(*PLAN_MRR[p]), 2) for p in plan_tier])
nps_missing = rng.random(N_CUSTOMERS) < 0.35
nps_score = np.where(nps_missing, np.nan, np.clip(rng.normal(35, 30, N_CUSTOMERS), -100, 100).round(0))

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": signup_dates,
    "plan_tier": plan_tier,
    "contract_type": contract_type,
    "region": region,
    "industry": industry,
    "initial_mrr": initial_mrr,
    "nps_score": nps_score,
})

# ------------------------------------------------------------------
# 2. USAGE_MONTHLY + churn simulation (month-by-month hazard model)
# ------------------------------------------------------------------
usage_rows = []
churn_month = {}   # customer_id -> churn month (Timestamp) or None if still active at END

for idx, row in customers.iterrows():
    cust = row["customer_id"]
    plan = row["plan_tier"]
    contract = row["contract_type"]
    signup = row["signup_date"]
    base_logins = {"Starter": 45, "Growth": 95, "Enterprise": 220}[plan]

    months_active = [m for m in ALL_MONTHS if m >= signup]
    engagement_trend = rng.normal(0, 1)   # customer-level baseline trend tendency
    tickets_tendency = rng.gamma(2.0, 1.2)

    cum_hazard_mult = 1.0
    consec_decline = 0
    prev_logins = None
    churned = False

    for m_i, m in enumerate(months_active):
        tenure_months = m_i
        # engagement decays slowly on average, with individual variation; a
        # "declining trend" streak raises churn risk realistically
        drift = rng.normal(-0.01 * (1 + engagement_trend * 0.3), 0.10)
        seasonal = 0.05 * np.sin(2 * np.pi * m.month / 12)
        logins = max(0, base_logins * (1 + drift * tenure_months / 12 + seasonal) * rng.uniform(0.75, 1.25))
        active_users = max(1, int(rng.poisson(lam={"Starter": 4, "Growth": 12, "Enterprise": 45}[plan])))
        features_used = int(np.clip(rng.poisson(lam={"Starter": 3, "Growth": 7, "Enterprise": 14}[plan]), 0, 25))
        support_tickets = int(rng.poisson(lam=max(0.1, tickets_tendency * (1 + 0.02 * tenure_months))))

        if prev_logins is not None and logins < prev_logins * 0.92:
            consec_decline += 1
        else:
            consec_decline = 0
        prev_logins = logins

        usage_rows.append({
            "customer_id": cust, "month": m.strftime("%Y-%m"), "active_users": active_users,
            "logins": round(logins, 1), "features_used": features_used, "support_tickets": support_tickets,
        })

        # ---- monthly churn hazard ----
        hazard = PLAN_BASE_HAZARD[plan] * CONTRACT_HAZARD_MULT[contract]
        if consec_decline >= 2:
            hazard *= 2.3
        if support_tickets >= 4:
            hazard *= 1.6
        if not np.isnan(row["nps_score"]) and row["nps_score"] < 0:
            hazard *= 1.5
        if tenure_months == 0:
            hazard *= 0.15  # brand new customers essentially never churn in month 0
        hazard = min(hazard, 0.35)

        if rng.random() < hazard and tenure_months >= 1:
            churned = True
            churn_month[cust] = m
            break

    if not churned:
        churn_month[cust] = None

usage_monthly = pd.DataFrame(usage_rows)

customers["churned"] = customers["customer_id"].map(lambda c: churn_month[c] is not None).astype(int)
customers["churn_month"] = customers["customer_id"].map(lambda c: churn_month[c].strftime("%Y-%m") if churn_month[c] is not None else None)

print(f"customers={len(customers)}  churn_rate={customers.churned.mean():.3f}  usage_rows={len(usage_monthly)}")
print(customers.groupby(["plan_tier","contract_type"]).churned.mean().round(3))

# ------------------------------------------------------------------
# save
# ------------------------------------------------------------------
customers.to_csv("customers.csv", index=False)
usage_monthly.to_csv("usage_monthly.csv", index=False)

conn = sqlite3.connect("churn.db")
customers.to_sql("customers", conn, if_exists="replace", index=False)
usage_monthly.to_sql("usage_monthly", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote CSVs + churn.db")
