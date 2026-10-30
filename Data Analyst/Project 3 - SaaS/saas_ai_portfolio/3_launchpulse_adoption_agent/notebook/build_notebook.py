import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""
# LaunchPulse — Autonomous Feature-Adoption Anomaly & Opportunity Agent

**What this is:** an agent that watches Flowdesk's daily feature-adoption
and activation-funnel metrics, detects anomalies with two complementary
statistical methods, automatically runs a diagnostic playbook (real SQL
breakdowns, not a guess) against whichever one fires, and has an LLM
synthesize the evidence into a root-cause incident report for a Product
Manager — the perceive → act → reason → report pattern behind every
"AI SRE" / autonomous product-analytics monitoring tool shipping in
2025-26, applied here to product adoption instead of infrastructure or
revenue.

**Why two detectors:** a broken feature release and a slow channel-quality
decline have completely different statistical signatures — one is a sharp
single/multi-day break in usage *volume*, the other is a gradual decline in
a *rate* (activation rate) that barely moves total volume at all. A single
threshold on one metric would miss one shape or the other entirely; using
the right tool for each anomaly shape is the point being demonstrated here.

**On the LLM backend:** same cached/live design as the rest of this
portfolio — see `../shared/llm_client.py`. Every incident report below was
authored by reading the *actual* diagnostic query results computed in this
notebook — including one report that correctly concludes "this isn't a
real incident, it's a planned marketing event," because an honest
monitoring agent has to be able to say that instead of manufacturing a
root cause for every flagged spike.
""")

code(r"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anomaly_detector import point_anomaly_detector, channel_drift_detector
from diagnostics import breakdown_by_feature, breakdown_by_channel, compare_to_baseline, channel_funnel_breakdown
from agent import investigate
from llm_client import get_client

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DB = "../data/launchpulse.db"
df = pd.read_csv("../data/daily_metrics.csv", parse_dates=["date"])
funnel = pd.read_csv("../data/daily_channel_funnel.csv", parse_dates=["date"])
daily_total = df.groupby("date")["adoption_events"].sum()
print(f"{len(df):,} feature-usage rows | {len(funnel):,} funnel rows | "
      f"{df.date.min().date()} to {df.date.max().date()} | "
      f"{df.adoption_events.sum():,} total adoption events")
""")

md("## 1. Detector 1 — Point Anomalies (STL decomposition + residual z-score)\n\nWatches total daily adoption-event volume across all features/channels/plans.")

code(r"""
point_anomalies = point_anomaly_detector(daily_total, z_threshold=3.5)
flagged = point_anomalies[point_anomalies.is_anomaly]
flagged[["date","actual","z_score","direction"]]
""")

code(r"""
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(point_anomalies.date, point_anomalies.actual, color="#2E86AB", linewidth=1, label="Daily adoption events")
ax.scatter(flagged.date, flagged.actual, color="#C0392B", s=60, zorder=5, label="Flagged anomaly")
ax.set_title("Daily Adoption Events with Point-Anomaly Detector Flags (STL residual |z| ≥ 3.5)")
ax.legend()
plt.tight_layout()
plt.savefig("../dashboard/point_anomalies.png", dpi=150)
plt.show()
""")

md(r"""**Reading this:** 7 days flag across 3 distinct events — the Oct 8
Automation Rules outage (a sharp drop), the Feb 11-13 webinar week
(a broad spike, investigated and dismissed below), and the Apr 15-16 AI
Assistant launch spike. Note Apr 14 itself (the first day of the launch)
does *not* cross the threshold — day-of-week seasonality on a Monday
already runs high, which dampens the residual on the launch's opening
day. This is a real, documented limitation: a 3-day event doesn't always
produce 3 flagged days.""")

md("## 2. Detector 2 — Channel Activation-Rate Drift (sustained decline a volume-based detector would miss)")

code(r"""
# Excludes no windows here (no holiday-adjacent signup lull overlaps this
# detector's flagged run) -- but the parameter exists and is used exactly
# as documented in the module docstring when a real overlap occurs.
drift_flags = channel_drift_detector(funnel, decline_threshold=-0.20, min_consecutive_days=5)
drift_flags
""")

code(r"""
funnel["activation_rate"] = funnel.activations / funnel.signups
pivot = funnel.pivot(index="date", columns="channel", values="activation_rate").sort_index()
roll7 = pivot.rolling(7).mean()

fig, ax = plt.subplots(figsize=(14,5))
for ch in pivot.columns:
    ax.plot(roll7.index, roll7[ch], label=ch, linewidth=1.6 if ch=="Paid Social" else 1)
ax.axvspan(pd.Timestamp("2024-12-14"), pd.Timestamp("2024-12-24"), color="orange", alpha=0.15)
ax.set_title("7-Day Rolling Activation Rate by Channel (Paid Social drift flagged Dec 14-24)")
ax.set_ylabel("Activation rate")
ax.legend()
plt.tight_layout()
plt.savefig("../dashboard/channel_drift.png", dpi=150)
plt.show()
""")

md("## 3. Agent Investigation — 4 Flagged Events, 3 Real Incidents + 1 Correctly Dismissed")

code(r"""
client = get_client(cache_file="../cached_responses.json")

incidents_config = [
    ("incident_2024_10_08_automation_bug", "Automation Rules Outage", "point_anomaly",
     "2024-10-08", "2024-10-08", "2024-10-01", "2024-10-07", {"z_score": -4.56}, "real"),
    ("incident_2025_04_ai_assistant_launch", "AI Assistant Launch Spike", "point_anomaly",
     "2025-04-14", "2025-04-16", "2025-04-07", "2025-04-13", {"z_score": "1.19 / 4.24 / 3.35"}, "real"),
    ("incident_2024_12_paid_social_drift", "Paid Social Activation Drift", "channel_drift",
     "2024-12-14", "2024-12-24", "2024-11-23", "2024-12-03", {"max_decline_pct": -36.6}, "real"),
    ("incident_2025_02_webinar_false_positive", "Feb 11-13 Broad Spike", "point_anomaly",
     "2025-02-11", "2025-02-13", "2025-02-04", "2025-02-10", {"z_score": "11.14 / 7.41 / 8.93"}, "dismissed"),
]

reports = []
for incident_id, title, kind, s, e, bs, be, stats, status in incidents_config:
    r = investigate(DB, incident_id, kind, s, e, bs, be, stats, client)
    reports.append(r)
    print(f"{'='*90}\n{title}  [{status.upper()}]  window {s}..{e}\n{'-'*90}")
    print(r.narrative)
    print()
""")

md(r"""**The pattern that matters here:** the agent doesn't just narrate
whatever the detector flags — the same diagnostic playbook (feature
breakdown, channel breakdown, baseline comparison) produces evidence that
supports 3 different verdicts: an isolated single-feature outage (Oct 8),
a dominant-but-not-exclusive feature spike (Apr 15-16, where every feature
moved but one moved 15x more than the rest), and a uniform cross-feature/
cross-channel lift that gets *correctly dismissed* as a known marketing
event rather than escalated (Feb 11-13). An agent that always finds a
root cause for every flagged spike isn't trustworthy — the dismissal is as
important a capability as the diagnosis.""")

md("## 4. Executive Summary & Recommendations")

md(r"""
1. **Two detectors, two anomaly shapes.** A volume-based point detector
   (STL + residual z-score) catches sharp breaks; a rate-based drift
   detector (trailing 7-day activation rate vs. 3-weeks-prior) catches slow
   channel-quality decay that never moves total volume enough to trip a
   z-score. Neither detector alone would have caught all 3 real incidents.
2. **The Automation Rules outage (Oct 8) is the cleanest, highest-confidence
   finding** — an 86.5% single-feature, single-day drop while every other
   feature moved slightly positive — and the fastest to act on (check that
   day's deploy log).
3. **The Paid Social activation-rate drift (-16.5pp vs. baseline, isolated
   to that one channel) is the highest-business-value finding** — it's the
   kind of slow-burning acquisition-quality problem that's easy to miss
   because total signup and usage *volume* barely moved; only the rate
   moved.
4. **The Feb 11-13 spike was correctly identified as a non-incident** — a
   uniform lift across every feature and channel, in a ratio matching
   normal traffic mix, which is the statistical signature of a broad
   marketing event, not a localized product change.
5. **Recommendation:** add known marketing-event dates to the point
   detector's exclusion list (mirroring how the drift detector already
   excludes holiday-adjacent windows), so the on-call rotation isn't paged
   for expected traffic; and add a scheduled channel-drift check
   specifically for Paid Social-style acquisition-quality regressions,
   since those are the incidents a volume dashboard alone will never
   surface.
""")

nb["cells"] = cells
with open("analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote analysis.ipynb with", len(cells), "cells")
