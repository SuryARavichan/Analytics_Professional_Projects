"""
Generates two related synthetic datasets for the Marketing A/B Test &
Attribution Analysis project:

1. ab_test_sessions.csv / .db table `ab_sessions`
   A 30-day checkout-redesign A/B test. Story baked in:
     - Overall lift is real but modest (~1.1-1.4pp on a ~9% baseline) --
       exactly the kind of result that needs a proper power/significance
       check, not a gut call.
     - The effect is NOT uniform: it's strong on Mobile, ~flat on Desktop
       (heterogeneous treatment effect) -- a segment cut a recruiter wants
       to see you catch instead of reporting one blended number.
     - Guardrail metric (page load time) is checked to make sure the new
       checkout didn't quietly hurt UX while helping conversion.

2. touchpoints.csv / .db table `touchpoints`  + channel_spend.csv
   Multi-touch marketing journeys for a separate cohort of converting and
   non-converting users across 5 channels, plus monthly channel spend, so
   first-touch / last-touch / linear / position-based attribution models
   can be compared and shown to produce meaningfully different "which
   channel gets the credit" answers -- the standard attribution-modeling
   talking point.
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

rng = np.random.default_rng(7)

# =====================================================================
# 1. A/B TEST: checkout redesign, 30-day test, randomized by session
# =====================================================================
TEST_START = datetime(2026, 3, 1)
TEST_DAYS = 30
N_SESSIONS = 48000

dates = [TEST_START + timedelta(days=int(d)) for d in rng.integers(0, TEST_DAYS, N_SESSIONS)]
device = rng.choice(["Mobile", "Desktop"], size=N_SESSIONS, p=[0.62, 0.38])
traffic_source = rng.choice(["Organic", "Paid Search", "Paid Social", "Direct", "Email"],
                             size=N_SESSIONS, p=[0.28, 0.24, 0.20, 0.16, 0.12])
variant = rng.choice(["control", "treatment"], size=N_SESSIONS, p=[0.5, 0.5])

# Baseline conversion by device
base_conv = np.where(device == "Mobile", 0.078, 0.112)
# Treatment effect: strong lift on mobile (+3.2pp relative ~ +2.6pp absolute),
# ~flat / slightly negative on desktop (-0.2pp) -> classic heterogeneous effect
treatment_lift = np.where(
    (variant == "treatment") & (device == "Mobile"), 0.026,
    np.where((variant == "treatment") & (device == "Desktop"), -0.002, 0.0)
)
conv_prob = np.clip(base_conv + treatment_lift + rng.normal(0, 0.004, N_SESSIONS), 0.01, 0.5)
converted = rng.binomial(1, conv_prob)

order_value = np.where(
    converted == 1,
    np.round(rng.gamma(shape=6.0, scale=12.5, size=N_SESSIONS), 2),
    0.0
)

# Guardrail metric: page load time (ms). New checkout (treatment) adds ~40ms
# median but well within an acceptable band -- a legitimate guardrail check,
# not a hidden problem.
base_load = rng.normal(1450, 220, N_SESSIONS)
load_time_ms = np.round(np.clip(base_load + np.where(variant == "treatment", 42, 0), 400, None), 0)

ab_sessions = pd.DataFrame({
    "session_id": [f"SESS{500000+i}" for i in range(N_SESSIONS)],
    "session_date": [d.date() for d in dates],
    "device": device,
    "traffic_source": traffic_source,
    "variant": variant,
    "converted": converted,
    "order_value": order_value,
    "page_load_time_ms": load_time_ms,
})
ab_sessions = ab_sessions.sort_values("session_date").reset_index(drop=True)

# =====================================================================
# 2. MULTI-TOUCH ATTRIBUTION: user journeys across channels
# =====================================================================
N_USERS = 9000
CHANNELS = ["Paid Search", "Paid Social", "Organic Search", "Email", "Referral"]
# Different channels play different roles in the funnel (assist vs closer)
CHANNEL_ROLE_WEIGHT = {  # relative likelihood of being LAST touch vs earlier touch
    "Paid Search": 1.4,
    "Paid Social": 0.6,   # great at awareness/assist, rarely closes
    "Organic Search": 1.1,
    "Email": 1.3,          # strong closer
    "Referral": 0.9,
}

journeys = []
touchpoint_rows = []
tp_id = 1
JOURNEY_START = datetime(2025, 10, 1)
JOURNEY_END = datetime(2026, 3, 31)

for u in range(N_USERS):
    user_id = f"USER{700000+u}"
    n_touches = rng.choice([1, 2, 3, 4, 5], p=[0.28, 0.27, 0.21, 0.14, 0.10])
    converts = rng.random() < 0.34  # overall conversion rate across journeys

    journey_start_offset = rng.integers(0, (JOURNEY_END - JOURNEY_START).days - 14)
    t0 = JOURNEY_START + timedelta(days=int(journey_start_offset))

    # sample channels with role weighting for the LAST slot
    channels_seq = list(rng.choice(CHANNELS, size=n_touches,
                                    p=np.array([1, 1, 1, 1, 1]) / 5))
    if converts:
        # bias the final touch toward higher "closer" weight channels
        weights = np.array([CHANNEL_ROLE_WEIGHT[c] for c in CHANNELS])
        weights = weights / weights.sum()
        channels_seq[-1] = str(rng.choice(CHANNELS, p=weights))

    touch_dates = sorted(t0 + timedelta(days=int(d)) for d in rng.integers(0, 14, n_touches))
    conv_value = round(float(rng.gamma(shape=5.5, scale=14.0)), 2) if converts else 0.0
    conv_date = touch_dates[-1] + timedelta(days=int(rng.integers(0, 3))) if converts else None

    for i, (ch, td) in enumerate(zip(channels_seq, touch_dates)):
        touchpoint_rows.append({
            "touchpoint_id": f"TP{800000+tp_id}",
            "user_id": user_id,
            "channel": ch,
            "touchpoint_date": td.date(),
            "touchpoint_order": i + 1,
            "touchpoints_in_journey": n_touches,
            "converted": int(converts),
            "conversion_date": conv_date.date() if converts else None,
            "conversion_value": conv_value if converts else 0.0,
        })
        tp_id += 1

touchpoints = pd.DataFrame(touchpoint_rows)

# =====================================================================
# 3. CHANNEL SPEND (monthly) -- for ROAS-by-attribution-model comparison
# =====================================================================
months = pd.date_range("2025-10-01", "2026-03-01", freq="MS")
spend_rows = []
BASE_SPEND = {"Paid Search": 42000, "Paid Social": 38000, "Organic Search": 6000,
              "Email": 4000, "Referral": 3000}
for m in months:
    for ch, base in BASE_SPEND.items():
        spend = round(base * rng.uniform(0.85, 1.15), 2)
        clicks = int(spend / rng.uniform(0.8, 3.2)) if ch in ("Paid Search", "Paid Social") else int(spend / rng.uniform(0.1, 0.4))
        spend_rows.append({"month": m.strftime("%Y-%m"), "channel": ch, "spend": spend, "clicks": clicks})
channel_spend = pd.DataFrame(spend_rows)

print(f"ab_sessions={len(ab_sessions)}  touchpoints={len(touchpoints)}  "
      f"converting_users={touchpoints[touchpoints.converted==1].user_id.nunique()}  "
      f"channel_spend_rows={len(channel_spend)}")

# ---- save ----
ab_sessions.to_csv("ab_test_sessions.csv", index=False)
touchpoints.to_csv("touchpoints.csv", index=False)
channel_spend.to_csv("channel_spend.csv", index=False)

conn = sqlite3.connect("marketing.db")
ab_sessions.to_sql("ab_sessions", conn, if_exists="replace", index=False)
touchpoints.to_sql("touchpoints", conn, if_exists="replace", index=False)
channel_spend.to_sql("channel_spend", conn, if_exists="replace", index=False)
conn.close()

print("Done. Wrote CSVs + marketing.db")
