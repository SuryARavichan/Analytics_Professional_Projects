"""
Generates 18,000 support tickets for Ravel AI, a fictional autonomous
AI customer-support-agent platform (the "agentic AI" trend: an LLM agent
that takes multi-step actions -- looks up an order, issues a refund,
resets a password -- and closes a ticket without a human, not just
answers a question).

THE CORE STORY, and why it's a genuinely different failure shape than
the prior portfolio's deploy-regression examples: Ravel ships an Agent
v2 policy update that lets the account_recovery agent take autonomous
account actions (password resets, security-override approvals) instead
of always escalating them to a human. The agent's OWN reported
containment rate for account_recovery jumps from 55% to 75% -- a
headline win. But a meaningful share of those autonomously-resolved
account_recovery tickets don't actually fix the customer's problem, so
customers reopen them within 48 hours far more often post-v2 (10% -> 35%
reopen rate). Once you adjust containment for reopens (TRUE containment,
not the agent's self-reported one), account_recovery barely moved at all
-- and it comes with real added cost, since a meaningful share of
reopened tickets eventually need a human anyway. This is deliberately a
sharper version of the "aggregate metric hides the real problem"
pattern used elsewhere in this session: here the deceptive metric isn't
just an average that hides a segment -- it's the AI SYSTEM'S OWN
SELF-REPORTED SUCCESS METRIC being gamed by a policy change, which is
exactly the kind of blind spot a real AgentOps analyst has to catch.

billing, technical, and shipping all get genuine (non-gamed) containment
improvements from the same v2 release, which is what makes the
account_recovery finding a real, isolated finding rather than "v2 is
just bad."
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(4242)

START = pd.Timestamp("2026-06-01")
N_DAYS = 60
DEPLOY_DAY = 30
DEPLOY_TS = START + pd.Timedelta(days=DEPLOY_DAY)
N_TICKETS = 18000

CATEGORIES = ["billing", "technical", "account_recovery", "shipping"]
CATEGORY_WEIGHT = {"billing": 0.30, "technical": 0.28, "account_recovery": 0.20, "shipping": 0.22}
CHANNELS = ["chat", "email"]
CHANNEL_WEIGHT = {"chat": 0.68, "email": 0.32}

# official (agent-self-reported) containment rate, by category and agent version
CONTAINMENT = {
    ("billing", "v1"): 0.72, ("billing", "v2"): 0.80,
    ("technical", "v1"): 0.60, ("technical", "v2"): 0.68,
    ("shipping", "v1"): 0.68, ("shipping", "v2"): 0.69,
    ("account_recovery", "v1"): 0.55, ("account_recovery", "v2"): 0.75,
}
# reopen-within-48h rate, AMONG agent-resolved tickets, by category and version
REOPEN_RATE = {
    ("billing", "v1"): 0.05, ("billing", "v2"): 0.05,
    ("technical", "v1"): 0.08, ("technical", "v2"): 0.07,
    ("shipping", "v1"): 0.06, ("shipping", "v2"): 0.06,
    ("account_recovery", "v1"): 0.10, ("account_recovery", "v2"): 0.35,
}
TOOL_CALLS_MEAN = {
    ("billing", "v1"): 1.6, ("billing", "v2"): 1.8,
    ("technical", "v1"): 3.2, ("technical", "v2"): 3.4,
    ("shipping", "v1"): 1.4, ("shipping", "v2"): 1.5,
    ("account_recovery", "v1"): 4.0, ("account_recovery", "v2"): 6.1,
}
RESOLUTION_TIME_MEAN_MIN = {"billing": 4.5, "technical": 9.0, "account_recovery": 6.5, "shipping": 3.5}

AGENT_COST_PER_TOOL_CALL = 0.05
AGENT_BASE_COST = 0.15
HUMAN_AGENT_COST_MEAN = 9.00
PROB_REOPEN_EVENTUALLY_ESCALATES = 0.70


def business_day_multiplier(dow: int) -> float:
    return 0.55 if dow >= 5 else 1.0


# ---- daily ticket volume, weekday-weighted ----
daily_base = N_TICKETS / N_DAYS
day_offsets = rng.choice(np.arange(N_DAYS), size=N_TICKETS,
                          p=(lambda w: w / w.sum())(np.array([business_day_multiplier((START + pd.Timedelta(days=d)).dayofweek) for d in range(N_DAYS)])))
timestamps = START + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(rng.integers(0, 24 * 3600, N_TICKETS), unit="s")

rows = []
for i in range(N_TICKETS):
    ts = timestamps[i]
    category = rng.choice(CATEGORIES, p=[CATEGORY_WEIGHT[c] for c in CATEGORIES])
    channel = rng.choice(CHANNELS, p=[CHANNEL_WEIGHT[c] for c in CHANNELS])
    version = "v2" if ts >= DEPLOY_TS else "v1"

    p_contained = np.clip(rng.normal(CONTAINMENT[(category, version)], 0.03), 0.05, 0.97)
    remaining = 1 - p_contained
    p_escalated = remaining * 0.8
    p_abandoned = remaining * 0.2
    resolution_path = rng.choice(
        ["agent_resolved", "escalated_to_human", "abandoned_by_customer"],
        p=[p_contained, p_escalated, p_abandoned],
    )

    n_tool_calls = max(0, int(round(rng.normal(TOOL_CALLS_MEAN[(category, version)],
                                                TOOL_CALLS_MEAN[(category, version)] * 0.35))))
    n_turns = max(1, int(round(rng.normal(3 + n_tool_calls * 0.8, 1.5))))
    resolution_time_min = max(0.5, rng.lognormal(np.log(RESOLUTION_TIME_MEAN_MIN[category]), 0.5))

    reopened = False
    if resolution_path == "agent_resolved":
        reopened = bool(rng.random() < REOPEN_RATE[(category, version)])

    # CSAT: only ~40% of tickets get a response; lower when reopened or escalated-then-abandoned-feeling
    csat_score = None
    if rng.random() < 0.40:
        if resolution_path == "agent_resolved" and not reopened:
            csat_score = int(np.clip(round(rng.normal(4.4, 0.6)), 1, 5))
        elif resolution_path == "agent_resolved" and reopened:
            csat_score = int(np.clip(round(rng.normal(2.3, 0.9)), 1, 5))
        elif resolution_path == "escalated_to_human":
            csat_score = int(np.clip(round(rng.normal(3.9, 0.8)), 1, 5))
        else:
            csat_score = int(np.clip(round(rng.normal(1.8, 0.7)), 1, 5))

    # cost: agent compute cost always incurred; escalation adds human cost;
    # a reopened "resolved" ticket often eventually needs a human anyway (hidden cost)
    agent_cost = AGENT_BASE_COST + n_tool_calls * AGENT_COST_PER_TOOL_CALL
    cost_usd = agent_cost
    if resolution_path == "escalated_to_human":
        cost_usd += max(1.0, rng.normal(HUMAN_AGENT_COST_MEAN, 2.0))
    elif resolution_path == "agent_resolved" and reopened and rng.random() < PROB_REOPEN_EVENTUALLY_ESCALATES:
        cost_usd += max(1.0, rng.normal(HUMAN_AGENT_COST_MEAN, 2.0))

    rows.append({
        "ticket_id": f"T{500000+i}", "timestamp": ts, "category": category, "channel": channel,
        "agent_version": version, "n_tool_calls": n_tool_calls, "n_turns": n_turns,
        "resolution_time_min": round(resolution_time_min, 2), "resolution_path": resolution_path,
        "agent_reported_resolved": resolution_path == "agent_resolved",
        "reopened_within_48h": reopened, "csat_score": csat_score, "cost_usd": round(cost_usd, 2),
    })

tickets = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
tickets.to_csv("tickets.csv", index=False)

print(f"{len(tickets):,} tickets | {tickets.timestamp.min()} to {tickets.timestamp.max()} | deploy at day {DEPLOY_DAY} ({DEPLOY_TS.date()})")
for version in ["v1", "v2"]:
    sub = tickets[tickets.agent_version == version]
    print(f"\n--- {version} ---")
    g = sub.groupby("category").apply(
        lambda d: pd.Series({
            "n": len(d),
            "official_containment": (d.resolution_path == "agent_resolved").mean(),
            "reopen_rate_of_contained": d.loc[d.resolution_path == "agent_resolved", "reopened_within_48h"].mean(),
        }), include_groups=False
    ).round(3)
    g["true_containment"] = (g.official_containment * (1 - g.reopen_rate_of_contained)).round(3)
    print(g)
print("\nWrote tickets.csv")
