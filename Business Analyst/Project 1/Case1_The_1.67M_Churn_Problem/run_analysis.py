"""
CASE 01 — Churn Intelligence & Retention Case
Loads the real IBM Telco Customer Churn dataset into SQLite, runs the
analysis queries in churn_analysis.sql, and produces chart images +
a results table for the report.
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette (dataviz skill reference palette) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

df = pd.read_csv("../data/telco_churn.csv")
# clean TotalCharges (blank strings for tenure==0 customers)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")

with open("churn_analysis.sql") as f:
    sql_text = f.read()

# split on ';' but keep queries with leading comments
queries = [q.strip() for q in sql_text.split(";") if q.strip() and "SELECT" in q.upper()]

results = {}
labels = [
    "01_overall_churn_rate", "02_churn_by_contract", "03_churn_by_tenure_bucket",
    "04_churn_by_payment_method", "05_churn_by_internet_service",
    "06_churn_by_support_bundle_fiber", "07_avg_monthly_charges_by_churn",
    "08_monthly_revenue_at_risk", "09_highest_risk_segment",
    "10_paperless_x_payment_top5",
]
for label, q in zip(labels, queries):
    res = pd.read_sql_query(q, conn)
    results[label] = res.to_dict(orient="records")
    print("\n===", label, "===")
    print(res.to_string(index=False))

with open("query_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------
# CHART 1 — Churn rate by contract type
# ---------------------------------------------------------------
d2 = pd.DataFrame(results["02_churn_by_contract"]).sort_values("churn_rate_pct", ascending=True)
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
bars = ax.barh(d2["Contract"], d2["churn_rate_pct"], color=[MUTED, MUTED, RED], height=0.55, zorder=3)
for b, v in zip(bars, d2["churn_rate_pct"]):
    ax.text(v + 0.8, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", fontsize=10, color=INK)
ax.set_xlim(0, max(d2["churn_rate_pct"]) * 1.25)
ax.set_title("Churn rate by contract type", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=10, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("chart_churn_by_contract.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — Churn rate by tenure bucket (line, since it's a progression)
# ---------------------------------------------------------------
d3 = pd.DataFrame(results["03_churn_by_tenure_bucket"])
order = ["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"]
d3["tenure_bucket"] = pd.Categorical(d3["tenure_bucket"], categories=order, ordered=True)
d3 = d3.sort_values("tenure_bucket")
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(d3["tenure_bucket"], d3["churn_rate_pct"], color=BLUE, linewidth=2.5, marker="o", markersize=6, zorder=3)
for x, y in zip(d3["tenure_bucket"], d3["churn_rate_pct"]):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, color=INK)
ax.set_ylim(0, max(d3["churn_rate_pct"]) * 1.3)
ax.set_title("Churn rate drops sharply after year one", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=9.5, colors=INK2)
ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("chart_churn_by_tenure.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 3 — Highest-risk segment vs base rate (comparison)
# ---------------------------------------------------------------
overall = results["01_overall_churn_rate"][0]["churn_rate_pct"]
seg = results["09_highest_risk_segment"][0]["churn_rate_pct"]
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
cats = ["All customers", "Month-to-month +\nFiber + no Tech Support"]
vals = [overall, seg]
bars = ax.bar(cats, vals, color=[MUTED, RED], width=0.5, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_ylim(0, max(vals)*1.3)
multiplier = seg / overall
ax.set_title(f"The highest-risk segment churns {multiplier:.1f}x the base rate", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=10, colors=INK2)
ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("chart_highest_risk_segment.png", facecolor=SURFACE)
plt.close()

print("\nOverall churn rate:", overall, "%")
print("Highest risk segment churn rate:", seg, "%")
print("Segment size:", results["09_highest_risk_segment"][0]["customers_in_segment"])
print("Monthly revenue at risk (all churned):", results["08_monthly_revenue_at_risk"][0]["monthly_revenue_lost"])
print("Annualized:", results["08_monthly_revenue_at_risk"][0]["annualized_revenue_lost"])
print("Charts saved.")
