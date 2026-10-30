"""
Generates 18 months (Jan 2024-Jun 2025) of daily revenue data for
"NimbusCart," a fictional online retailer, broken out by region (4) and
channel (3) -- the dimensional breakdown SentinelBI's diagnostic agent
queries when investigating a flagged anomaly.

Three distinct, realistic incidents are injected:
  1. POINT ANOMALY — a payment-gateway outage on 2024-09-12 that craters
     Web-channel revenue for the day (other channels ~unaffected).
  2. POINT ANOMALY (positive) — a 3-day regional promo spike in the West
     region around Black Friday/Cyber Monday (2024-11-29 to 12-01).
  3. LEVEL-SHIFT ANOMALY — a slow ~3-week decline in the South region
     (2025-02-01 to 2025-02-21), simulating a competitor launch -- too
     gradual for a daily point-anomaly detector to catch cleanly, which is
     the point: it needs a different detection method (see
     pipeline/anomaly_detector.py's two detectors).
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

rng = np.random.default_rng(55)

START = datetime(2024, 1, 1)
END = datetime(2025, 6, 30)
DATES = pd.date_range(START, END, freq="D")

REGIONS = ["Northeast", "Midwest", "South", "West"]
REGION_BASE_SHARE = {"Northeast": 0.27, "Midwest": 0.22, "South": 0.29, "West": 0.22}
CHANNELS = ["Web", "Mobile App", "Marketplace"]
CHANNEL_SHARE = {"Web": 0.46, "Mobile App": 0.34, "Marketplace": 0.20}

BASE_DAILY_REVENUE = 148000

rows = []
for d in DATES:
    # weekly seasonality (weekend lift for a consumer retailer) + slow annual growth trend + holiday bump
    dow_mult = {0: 0.92, 1: 0.90, 2: 0.93, 3: 0.97, 4: 1.08, 5: 1.22, 6: 1.15}[d.weekday()]
    trend_mult = 1 + 0.00035 * (d - START).days  # slow organic growth over 18 months
    month_mult = 1.0
    if d.month == 12:
        month_mult = 1.35 if d.day <= 24 else 0.85
    elif d.month == 11 and d.day >= 20:
        month_mult = 1.25
    elif d.month in (1,):
        month_mult = 0.92

    day_total = BASE_DAILY_REVENUE * dow_mult * trend_mult * month_mult * rng.lognormal(0, 0.05)

    for region in REGIONS:
        r_share = REGION_BASE_SHARE[region]
        for channel in CHANNELS:
            c_share = CHANNEL_SHARE[channel]
            revenue = day_total * r_share * c_share * rng.lognormal(0, 0.06)

            # ---- Incident 1: payment gateway outage, Web channel, single day ----
            if d.date() == datetime(2024, 9, 12).date() and channel == "Web":
                revenue *= 0.22

            # ---- Incident 2: West region promo spike, 3 days ----
            if datetime(2024,11,29).date() <= d.date() <= datetime(2024,12,1).date() and region == "West":
                revenue *= 2.65

            # ---- Incident 3: South region gradual decline, ~3 weeks ----
            if datetime(2025,2,1).date() <= d.date() <= datetime(2025,2,21).date() and region == "South":
                days_in = (d.date() - datetime(2025,2,1).date()).days
                decline_mult = 1 - min(0.38, 0.02 * days_in)  # ramps down to ~-38% by day 19, holds
                revenue *= decline_mult

            orders = max(1, int(revenue / rng.uniform(58, 78)))
            sessions = int(orders / rng.uniform(0.018, 0.028))

            rows.append({
                "date": d.date(), "region": region, "channel": channel,
                "revenue": round(revenue, 2), "orders": orders, "sessions": sessions,
            })

daily_metrics = pd.DataFrame(rows)
print(f"rows={len(daily_metrics)}  date_range={daily_metrics.date.min()}..{daily_metrics.date.max()}  "
      f"total_revenue=${daily_metrics.revenue.sum():,.0f}")

daily_metrics.to_csv("daily_metrics.csv", index=False)
conn = sqlite3.connect("nimbuscart.db")
daily_metrics.to_sql("daily_metrics", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote daily_metrics.csv + nimbuscart.db")
