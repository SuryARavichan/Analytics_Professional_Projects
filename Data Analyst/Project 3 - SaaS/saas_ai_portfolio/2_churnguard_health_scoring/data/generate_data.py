"""
Generates a B2B SaaS customer-health dataset for ChurnGuard AI: 2,800
accounts with 24 months of monthly usage history, seat licensing/
utilization, support signal, and simulated churn AND expansion outcomes --
this is a two-sided dataset (down-risk and up-opportunity), not just a
churn dataset, because a real health score has to serve both a "save this
account" motion and a "sell this account more" motion.

Story baked in:
  - Seat utilization (active_seats / seat_count) is the key signal for
    BOTH outcomes: chronically low utilization predicts churn; utilization
    consistently near/at the seat cap predicts expansion (they need more
    seats).
  - Monthly contracts churn more than Annual (as in the earlier churn
    project) -- kept for realism, but now paired with the expansion side.
  - Support ticket sentiment (positive/negative resolution) affects churn
    risk independent of usage.
  - Every account has a next_renewal_date, so the notebook can prioritize
    "who does CS need to prep for in the next 30/60/90 days," a genuinely
    operational SaaS CS workflow question.
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

rng = np.random.default_rng(88)

N_ACCOUNTS = 2800
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
ALL_MONTHS = pd.date_range(START, END, freq="MS")
TODAY = pd.Timestamp("2025-12-31")  # "as of" date for renewal-window calcs

PLAN_TIERS = ["Starter", "Growth", "Enterprise"]
PLAN_W = [0.42, 0.39, 0.19]
PLAN_SEAT_RANGE = {"Starter": (2, 15), "Growth": (10, 60), "Enterprise": (40, 300)}
PLAN_PRICE_PER_SEAT = {"Starter": 15, "Growth": 22, "Enterprise": 34}
PLAN_BASE_HAZARD = {"Starter": 0.022, "Growth": 0.011, "Enterprise": 0.005}

CONTRACT_TYPES = ["Monthly", "Annual"]
CONTRACT_W = [0.50, 0.50]
CONTRACT_HAZARD_MULT = {"Monthly": 1.55, "Annual": 0.42}
RENEWAL_CADENCE_MONTHS = {"Monthly": 1, "Annual": 12}

INDUSTRIES = ["Software", "Financial Services", "Healthcare", "Retail", "Manufacturing", "Media", "Education"]
REGIONS = ["North America", "EMEA", "APAC"]

# ------------------------------------------------------------------
# 1. ACCOUNTS
# ------------------------------------------------------------------
account_ids = [f"ACCT{60000+i}" for i in range(N_ACCOUNTS)]
plan_tier = rng.choice(PLAN_TIERS, size=N_ACCOUNTS, p=PLAN_W)
contract_type = rng.choice(CONTRACT_TYPES, size=N_ACCOUNTS, p=CONTRACT_W)
industry = rng.choice(INDUSTRIES, size=N_ACCOUNTS)
region = rng.choice(REGIONS, size=N_ACCOUNTS, p=[0.55, 0.30, 0.15])
seat_count = np.array([int(rng.integers(*PLAN_SEAT_RANGE[p])) for p in plan_tier])
mrr = np.round(seat_count * np.array([PLAN_PRICE_PER_SEAT[p] for p in plan_tier]) * rng.uniform(0.9, 1.1, N_ACCOUNTS), 2)

signup_month_idx = rng.integers(0, 20, N_ACCOUNTS)
signup_dates = [ALL_MONTHS[i] for i in signup_month_idx]
nps_missing = rng.random(N_ACCOUNTS) < 0.30
nps_score = np.where(nps_missing, np.nan, np.clip(rng.normal(32, 32, N_ACCOUNTS), -100, 100).round(0))

accounts = pd.DataFrame({
    "account_id": account_ids, "plan_tier": plan_tier, "contract_type": contract_type,
    "industry": industry, "region": region, "seat_count": seat_count, "mrr": mrr,
    "signup_date": signup_dates, "nps_score": nps_score,
})

# ------------------------------------------------------------------
# 2. MONTHLY USAGE + support + churn/expansion simulation
# ------------------------------------------------------------------
usage_rows = []
churn_month, expanded_flag, expansion_month, next_renewal = {}, {}, {}, {}

for idx, row in accounts.iterrows():
    acct = row["account_id"]
    plan = row["plan_tier"]
    contract = row["contract_type"]
    signup = row["signup_date"]
    seats = row["seat_count"]

    # baseline utilization tendency per account: some run lean (low util),
    # some run hot (near/at cap, the expansion-prone ones)
    util_tendency = np.clip(rng.beta(2.2, 2.5), 0.05, 0.99)
    util_trend = rng.normal(0.004, 0.01)  # slow drift, positive = growing usage
    tickets_tendency = rng.gamma(1.6, 1.0)

    months_active = [m for m in ALL_MONTHS if m >= signup]
    consec_low_util_months = 0
    churned = False
    will_expand = False

    for m_i, m in enumerate(months_active):
        tenure = m_i
        util = np.clip(util_tendency + util_trend * tenure + rng.normal(0, 0.05), 0.03, 1.0)
        active_seats = max(1, int(round(util * seats)))
        support_tickets = int(rng.poisson(lam=max(0.05, tickets_tendency * (1 + 0.01*tenure))))
        ticket_negative = int(rng.random() < 0.35) if support_tickets > 0 else 0

        usage_rows.append({
            "account_id": acct, "month": m.strftime("%Y-%m"), "seat_count": seats,
            "active_seats": active_seats, "utilization_pct": round(100*active_seats/seats, 1),
            "support_tickets": support_tickets, "ticket_negative_sentiment": ticket_negative,
        })

        if active_seats / seats < 0.35:
            consec_low_util_months += 1
        else:
            consec_low_util_months = 0

        # ---- churn hazard ----
        hazard = PLAN_BASE_HAZARD[plan] * CONTRACT_HAZARD_MULT[contract]
        if consec_low_util_months >= 2:
            hazard *= 2.4
        if support_tickets >= 3 and ticket_negative:
            hazard *= 1.7
        if not np.isnan(row["nps_score"]) and row["nps_score"] < 0:
            hazard *= 1.5
        if tenure == 0:
            hazard *= 0.1
        hazard = min(hazard, 0.30)

        if rng.random() < hazard and tenure >= 1:
            churned = True
            churn_month[acct] = m
            break

        # ---- expansion trigger: sustained high utilization (>=85%) for 3+ consecutive months ----
        if util >= 0.80 and tenure >= 2 and not will_expand:
            # check last 2 recorded rows for this account also high (approx via util_trend positive & high util_tendency)
            if util_tendency > 0.55 and util_trend > -0.001 and rng.random() < 0.40:
                will_expand = True
                expansion_month[acct] = m

    if not churned:
        churn_month[acct] = None
    expanded_flag[acct] = will_expand
    if will_expand and acct not in expansion_month:
        expansion_month[acct] = None

    # next renewal date: based on contract cadence from signup, first renewal on/after TODAY (only meaningful for non-churned)
    cadence = RENEWAL_CADENCE_MONTHS[contract]
    renewal = pd.Timestamp(signup)
    while renewal <= TODAY:
        renewal = renewal + pd.DateOffset(months=cadence)
    next_renewal[acct] = renewal.date()

usage_monthly = pd.DataFrame(usage_rows)
accounts["churned"] = accounts["account_id"].map(lambda a: churn_month[a] is not None).astype(int)
accounts["churn_month"] = accounts["account_id"].map(lambda a: churn_month[a].strftime("%Y-%m") if churn_month[a] else None)
accounts["expanded"] = accounts["account_id"].map(lambda a: int(expanded_flag[a]))
accounts["expansion_month"] = accounts["account_id"].map(lambda a: expansion_month[a].strftime("%Y-%m") if expansion_month.get(a) else None)
accounts["next_renewal_date"] = accounts["account_id"].map(next_renewal)
accounts["days_to_renewal"] = (pd.to_datetime(accounts["next_renewal_date"]) - TODAY).dt.days

print(f"accounts={len(accounts)}  churn_rate={accounts.churned.mean():.3f}  expansion_rate={accounts.expanded.mean():.3f}")
print(accounts.groupby(["plan_tier","contract_type"]).churned.mean().round(3))

accounts.to_csv("accounts.csv", index=False)
usage_monthly.to_csv("usage_monthly.csv", index=False)

conn = sqlite3.connect("churnguard.db")
accounts.to_sql("accounts", conn, if_exists="replace", index=False)
usage_monthly.to_sql("usage_monthly", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote CSVs + churnguard.db")
