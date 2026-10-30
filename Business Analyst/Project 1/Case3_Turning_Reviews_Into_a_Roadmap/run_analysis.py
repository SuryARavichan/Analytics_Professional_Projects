"""
CASE 03 — Feature Backlog From the Wild
Real Google Play "ChatGPT" app reviews. Rule-based theme mining (validated
against a manual read, same as the case study's methodology), turned into
a RICE-prioritized, Jira-importable backlog + a simulated 2-sprint velocity view.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import re

plt.rcParams["font.family"] = "DejaVu Sans"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GOOD = "#0ca30c"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

r = pd.read_csv("../data/playstore_reviews.csv")
sub = r[r["app_title"] == "ChatGPT"].copy()
sub["review"] = sub["review"].astype(str)
sub["len"] = sub["review"].str.len()
pool = sub[(sub["review_rating"] <= 3) | (sub["len"] >= 25)].reset_index(drop=True)

themes = {
    "App stability / crashes / bugs": r"\bglitch|\bbug\b|crash|\bbroke\b|breaks\b|freeze|\bstuck\b|not working|won'?t load|error\b",
    "Memory / context loss across chats": r"\bforget(s|ful)?\b|doesn'?t remember|won'?t remember|contradict|repeats? errors?.{0,15}chat|overfill",
    "Accuracy / wrong or inconsistent answers": r"wrong (information|diagram|answer)|inaccurate|misleading|hallucinat|doesn'?t listen|ignored my|not listening",
    "Slow performance / response lag": r"\bslow\b|takes? (too )?(long|time)|\blag\b|time le raha",
    "Paywall / upgrade blocks core features": r"money hungry|paywall|can'?t.{0,20}without upgrad|upgrad(e|ing).{0,20}(photo|image|feature)|subscription.{0,15}(force|require)",
    "Forced sign-in / account friction": r"forced sign|force.{0,10}sign|sign[- ]?up.{0,15}(can'?t|won'?t|unable)|can'?t.{0,15}sign[- ]?up",
}

pool["themes"] = [[] for _ in range(len(pool))]
for name, pattern in themes.items():
    mask = pool["review"].str.contains(pattern, case=False, regex=True, na=False)
    for idx in pool[mask].index:
        pool.at[idx, "themes"].append(name)

counts = {}
for lst in pool["themes"]:
    for t in lst:
        counts[t] = counts.get(t, 0) + 1

tagged_n = (pool["themes"].apply(len) > 0).sum()
summary = {
    "pool_size": int(len(pool)),
    "tagged_reviews": int(tagged_n),
    "theme_counts": counts,
}
with open("theme_mining_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# ---------------------------------------------------------------
# CHART — theme frequency
# ---------------------------------------------------------------
s = pd.Series(counts).sort_values(ascending=True)
colors = [RED if i == len(s)-1 else MUTED for i in range(len(s))]
fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.barh(s.index, s.values, color=colors, height=0.55, zorder=3)
for b, v in zip(bars, s.values):
    ax.text(v + 0.15, b.get_y()+b.get_height()/2, f"{v}", va="center", fontsize=9.5, color=INK)
ax.set_title(f"Reliability complaints outnumber every other theme\n(n={summary['tagged_reviews']} tagged mentions in {summary['pool_size']} reviews screened)",
             fontsize=11.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("chart_theme_frequency.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# RICE backlog
# ---------------------------------------------------------------
backlog = [
    dict(epic="Reliability & Bug Triage", story="Retry a failed message send without duplicating it",
         reach=8, impact=3, confidence=90, effort=3, sprint=1, points=5),
    dict(epic="Reliability & Bug Triage", story="Show a clear error state (network vs server vs input) instead of a silent failure",
         reach=8, impact=2, confidence=80, effort=2, sprint=1, points=3),
    dict(epic="Conversation Memory & Consistency", story="Warn the user before a long chat's context window truncates",
         reach=5, impact=3, confidence=70, effort=5, sprint=1, points=8),
    dict(epic="Conversation Memory & Consistency", story="Keep the fix to a corrected error from being repeated later in the same chat",
         reach=5, impact=2, confidence=60, effort=5, sprint=2, points=8),
    dict(epic="Answer Accuracy & Trust Signals", story="Flag lower-confidence answers on factual / diagram-based questions",
         reach=6, impact=3, confidence=50, effort=8, sprint=2, points=13),
    dict(epic="Transparent Paywall & Usage Limits", story="Show which features require upgrading before the user attempts them, not after",
         reach=4, impact=2, confidence=85, effort=2, sprint=1, points=3),
    dict(epic="Frictionless Onboarding", story="Let a new user send their first message before requiring account sign-in",
         reach=7, impact=2, confidence=60, effort=5, sprint=2, points=8),
]
bdf = pd.DataFrame(backlog)
bdf["rice_score"] = (bdf["reach"] * bdf["impact"] * (bdf["confidence"]/100) / bdf["effort"]).round(1)
bdf = bdf.sort_values("rice_score", ascending=False).reset_index(drop=True)
bdf.insert(0, "rank", bdf.index + 1)
bdf.to_csv("rice_backlog.csv", index=False)
print("\nRICE-ranked backlog:")
print(bdf[["rank","epic","story","reach","impact","confidence","effort","rice_score","sprint","points"]].to_string(index=False))

# ---------------------------------------------------------------
# Jira-importable CSV  (Jira CSV importer field names)
# ---------------------------------------------------------------
jira_rows = []
epic_keys = {e: f"BACK-{i+1}" for i, e in enumerate(bdf["epic"].unique())}
for e, key in epic_keys.items():
    jira_rows.append({
        "Issue Type": "Epic", "Summary": e, "Epic Name": e, "Description": "",
        "Priority": "", "Story Points": "", "Epic Link": "", "Sprint": "",
    })
for _, row in bdf.iterrows():
    jira_rows.append({
        "Issue Type": "Story",
        "Summary": row["story"],
        "Epic Name": "",
        "Description": f"As a ChatGPT app user, I want this fixed/added so the reliability issue behind it (RICE {row['rice_score']}) stops recurring.",
        "Priority": "High" if row["rank"] <= 2 else ("Medium" if row["rank"] <= 5 else "Low"),
        "Story Points": row["points"],
        "Epic Link": row["epic"],
        "Sprint": f"Sprint {row['sprint']}",
    })
jira_df = pd.DataFrame(jira_rows)
jira_df.to_csv("jira_import_backlog.csv", index=False)
print("\nJira-importable CSV written:", len(jira_df), "rows")

# ---------------------------------------------------------------
# Simulated 2-sprint velocity chart
# ---------------------------------------------------------------
sprint1_committed = int(bdf[bdf["sprint"]==1]["points"].sum())
sprint2_committed = int(bdf[bdf["sprint"]==2]["points"].sum())
sprint1_completed = sprint1_committed - 3   # simulated: one story slipped
sprint2_completed = sprint2_committed        # simulated: fully cleared

fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = ["Sprint 1", "Sprint 2"]
committed = [sprint1_committed, sprint2_committed]
completed = [sprint1_completed, sprint2_completed]
w = 0.32
xpos = range(len(x))
b1 = ax.bar([p - w/2 for p in xpos], committed, width=w, color=MUTED, label="Committed", zorder=3)
b2 = ax.bar([p + w/2 for p in xpos], completed, width=w, color=AQUA, label="Completed", zorder=3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f"{int(b.get_height())}", ha="center", fontsize=9.5, color=INK)
ax.set_xticks(list(xpos)); ax.set_xticklabels(x, fontsize=10, color=INK2)
ax.set_title("Sprint velocity (simulated execution)", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK2)
plt.tight_layout()
plt.savefig("chart_sprint_velocity.png", facecolor=SURFACE)
plt.close()

print(f"\nSprint 1: committed {sprint1_committed} / completed {sprint1_completed}")
print(f"Sprint 2: committed {sprint2_committed} / completed {sprint2_completed}")
print("Charts and CSVs saved.")
