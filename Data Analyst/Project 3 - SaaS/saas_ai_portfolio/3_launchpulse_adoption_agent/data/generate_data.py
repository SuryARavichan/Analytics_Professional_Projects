"""
Generates 18 months (Jan 2024-Jun 2025) of daily product-usage data for
"Flowdesk," the same fictional SaaS product used in PathfinderAI, this time
seen through a feature-adoption / activation-funnel lens instead of a
whole-database NL-query lens.

Two tables:
  daily_metrics          — date x feature_name x channel x plan_tier grain,
                            adoption_events (feature_used events) + active
                            accounts. The headline series LaunchPulse's
                            point-anomaly detector watches is the daily
                            total across all 96 combinations.
  daily_channel_funnel    — date x channel grain, signups + activations,
                            the series the drift detector watches
                            (activation_rate = activations/signups),
                            because a channel-level activation problem
                            shows up in the funnel, not in feature-usage
                            volume.

Four injected incidents (deliberately mirroring SentinelBI's structure —
point/point/drift/false-positive — but for a product-analytics domain):
  1. POINT ANOMALY (negative) — a broken release craters "Automation
     Rules" feature usage for a single day (2024-10-08).
  2. POINT ANOMALY (positive) — a product-launch spike in "AI Assistant"
     usage across all channels for 3 days (2025-04-14 to 04-16).
  3. DRIFT ANOMALY — a ~3-week gradual activation-rate decline in the
     Paid Social channel (2024-12-01 to 2024-12-21), simulating a
     landing-page/onboarding mismatch after a campaign change -- too
     gradual for the point detector, which is exactly why a second
     detector exists.
  4. FALSE POSITIVE (correctly dismissed) — a planned marketing webinar
     week (2025-02-11 to 02-13) drives a broad, cross-feature usage spike
     that trips the point detector but is not an incident.
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

rng = np.random.default_rng(77)

START = datetime(2024, 1, 1)
END = datetime(2025, 6, 30)
DATES = pd.date_range(START, END, freq="D")

FEATURES = ["Automation Rules", "AI Assistant", "Custom Dashboards", "Bulk Import",
            "Team Workspaces", "API Access", "Advanced Reporting", "Slack Integration"]
FEATURE_SHARE = {
    "Automation Rules": 0.19, "AI Assistant": 0.09, "Custom Dashboards": 0.16,
    "Bulk Import": 0.08, "Team Workspaces": 0.14, "API Access": 0.07,
    "Advanced Reporting": 0.15, "Slack Integration": 0.12,
}
CHANNELS = ["Organic Search", "Paid Search", "Paid Social", "Referral"]
CHANNEL_SHARE = {"Organic Search": 0.38, "Paid Search": 0.24, "Paid Social": 0.21, "Referral": 0.17}
PLAN_TIERS = ["Starter", "Growth", "Enterprise"]
PLAN_SHARE = {"Starter": 0.45, "Growth": 0.37, "Enterprise": 0.18}

BASE_DAILY_EVENTS = 15400

rows = []
funnel_rows = []
for d in DATES:
    # B2B usage pattern: weekday-heavy, sharp weekend drop-off (opposite of consumer retail)
    dow_mult = {0: 1.05, 1: 1.10, 2: 1.12, 3: 1.08, 4: 0.95, 5: 0.55, 6: 0.50}[d.weekday()]
    trend_mult = 1 + 0.00060 * (d - START).days  # steady product-usage growth over 18 months
    month_mult = 1.0
    if d.month == 12 and d.day >= 20:
        month_mult = 0.68   # holiday lull
    elif d.month == 1 and d.day <= 7:
        month_mult = 0.80   # slow return from holidays
    elif d.month == 7 and d.day <= 7:
        month_mult = 0.90   # July 4th week lull

    # ---- Incident 4: planned webinar week -- broad, cross-feature, cross-channel spike ----
    webinar_mult = 1.55 if datetime(2025, 2, 11).date() <= d.date() <= datetime(2025, 2, 13).date() else 1.0

    day_total = BASE_DAILY_EVENTS * dow_mult * trend_mult * month_mult * webinar_mult * rng.lognormal(0, 0.05)

    for feature in FEATURES:
        f_share = FEATURE_SHARE[feature]
        for channel in CHANNELS:
            c_share = CHANNEL_SHARE[channel]
            for plan in PLAN_TIERS:
                p_share = PLAN_SHARE[plan]
                events = day_total * f_share * c_share * p_share * rng.lognormal(0, 0.10)

                # ---- Incident 1: Automation Rules broken release, single day ----
                if feature == "Automation Rules" and d.date() == datetime(2024, 10, 8).date():
                    events *= 0.12

                # ---- Incident 2: AI Assistant launch spike, 3 days ----
                if feature == "AI Assistant" and datetime(2025, 4, 14).date() <= d.date() <= datetime(2025, 4, 16).date():
                    events *= 3.4

                active_accounts = max(1, int(events / rng.uniform(1.6, 2.4)))
                rows.append({
                    "date": d.date(), "feature_name": feature, "channel": channel, "plan_tier": plan,
                    "adoption_events": max(0, int(round(events))), "active_accounts": active_accounts,
                })

    # ---- channel-level daily funnel (signups -> activations) -- feeds the drift detector ----
    for channel in CHANNELS:
        c_share = CHANNEL_SHARE[channel]
        base_signups = 210 * c_share / 0.25
        signups = max(0, int(base_signups * dow_mult * trend_mult * month_mult * rng.lognormal(0, 0.12)))
        base_activation_rate = 0.58

        # ---- Incident 3: Paid Social activation-rate drift, ~3 weeks ----
        if channel == "Paid Social" and datetime(2024, 12, 1).date() <= d.date() <= datetime(2024, 12, 21).date():
            days_in = (d.date() - datetime(2024, 12, 1).date()).days
            base_activation_rate = 0.58 * (1 - min(0.40, 0.021 * days_in))

        activation_rate = float(np.clip(base_activation_rate * rng.lognormal(0, 0.05), 0.05, 0.95))
        activations = int(round(signups * activation_rate))
        funnel_rows.append({"date": d.date(), "channel": channel, "signups": signups, "activations": activations})

daily_metrics = pd.DataFrame(rows)
daily_channel_funnel = pd.DataFrame(funnel_rows)

print(f"daily_metrics: rows={len(daily_metrics):,}  date_range={daily_metrics.date.min()}..{daily_metrics.date.max()}  "
      f"total_adoption_events={daily_metrics.adoption_events.sum():,}")
print(f"daily_channel_funnel: rows={len(daily_channel_funnel):,}  "
      f"total_signups={daily_channel_funnel.signups.sum():,}  total_activations={daily_channel_funnel.activations.sum():,}  "
      f"overall_activation_rate={daily_channel_funnel.activations.sum()/daily_channel_funnel.signups.sum():.1%}")

daily_metrics.to_csv("daily_metrics.csv", index=False)
daily_channel_funnel.to_csv("daily_channel_funnel.csv", index=False)
conn = sqlite3.connect("launchpulse.db")
daily_metrics.to_sql("daily_metrics", conn, if_exists="replace", index=False)
daily_channel_funnel.to_sql("daily_channel_funnel", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote daily_metrics.csv, daily_channel_funnel.csv + launchpulse.db")
