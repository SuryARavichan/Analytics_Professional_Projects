"""
CASE 02 — Debt Verification & Dispute Intake Process Overhaul
Real CFPB consumer complaint extract (26k complaints, Debt collection product).
Produces charts + a current/future-state process diagram.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json

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

c = pd.read_csv("../data/consumer_complaints.csv")
dc = c[c["Product"] == "Debt collection"].copy()

issue_counts = dc["Issue"].value_counts()
top_issue = issue_counts.idxmax()
sub = dc[dc["Issue"] == top_issue]
subissue_counts = sub["Sub-issue"].value_counts()

results = {
    "total_debt_collection_complaints": int(len(dc)),
    "top_issue": top_issue,
    "top_issue_count": int(issue_counts.max()),
    "top_issue_pct_of_all": round(100 * issue_counts.max() / len(dc), 1),
    "subissue_breakdown": subissue_counts.to_dict(),
    "debt_not_mine_count": int(subissue_counts.get("Debt is not mine", 0)),
    "debt_not_mine_pct_of_top_issue": round(100 * subissue_counts.get("Debt is not mine", 0) / len(sub), 1),
    "untimely_response_pct_top_issue": round(100 * (sub["Timely response?"] == "No").sum() / len(sub), 2),
    "issue_counts": issue_counts.head(6).to_dict(),
}
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------
# CHART 1 — Complaint volume by issue
# ---------------------------------------------------------------
top6 = issue_counts.head(6).sort_values(ascending=True)
colors = [MUTED]*5 + [RED]
colors = [RED if i == len(top6)-1 else MUTED for i in range(len(top6))]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
labels = [l if len(l) < 34 else l[:31]+"..." for l in top6.index]
bars = ax.barh(labels, top6.values, color=colors, height=0.6, zorder=3)
for b, v in zip(bars, top6.values):
    ax.text(v + 30, b.get_y()+b.get_height()/2, f"{v:,}", va="center", fontsize=9.5, color=INK)
ax.set_title("\"Debt not owed\" dwarfs every other complaint type", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("chart_complaints_by_issue.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — Sub-issue breakdown of the top issue (root cause)
# ---------------------------------------------------------------
sc = subissue_counts.sort_values(ascending=True)
colors2 = [RED if i == len(sc)-1 else MUTED for i in range(len(sc))]
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.barh(sc.index, sc.values, color=colors2, height=0.55, zorder=3)
total = sc.sum()
for b, v in zip(bars, sc.values):
    ax.text(v + 25, b.get_y()+b.get_height()/2, f"{v:,} ({100*v/total:.0f}%)", va="center", fontsize=9.5, color=INK)
ax.set_title(f'Root cause: {results["debt_not_mine_pct_of_top_issue"]:.0f}% say "this debt isn\'t mine"', fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_xlim(0, max(sc.values)*1.32)
plt.tight_layout()
plt.savefig("chart_root_cause_subissue.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# DIAGRAM — Current-state vs future-state process map
# ---------------------------------------------------------------
def box(ax, xy, w, h, text, fc, ec, tc=INK, fs=9.2, weight="normal"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc,
            weight=weight, wrap=True, zorder=4)

def arrow(ax, p1, p2, color=MUTED):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                         color=color, linewidth=1.4, zorder=2)
    ax.add_patch(a)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)

# --- current state ---
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.set_facecolor(SURFACE)
ax.set_title("CURRENT STATE — no verification gate", fontsize=11.5, color=RED, loc="left", weight="bold", pad=10)
steps_cs = [
    (0.3, 4.4, 2.1, 1.1, "Account placed\nwith collector"),
    (2.9, 4.4, 2.1, 1.1, "Collector\ncontacts consumer"),
    (5.5, 4.4, 2.3, 1.1, "Consumer disputes:\n“not my debt”"),
    (8.3, 4.4, 1.4, 1.1, "CFPB\ncomplaint"),
]
for x, y, w, h, t in steps_cs:
    box(ax, (x, y), w, h, t, "#ffffff", GRID, fs=8.6)
for i in range(len(steps_cs)-1):
    x1 = steps_cs[i][0] + steps_cs[i][2]
    x2 = steps_cs[i+1][0]
    y = steps_cs[i][1] + steps_cs[i][3]/2
    arrow(ax, (x1, y), (x2, y))
box(ax, (2.4, 1.4), 5.2, 1.1, "63% of “not owed” complaints trace\nback here — no identity check before contact", "#fbe7e6", RED, tc="#7a1f1f", fs=8.8, weight="bold")
arrow(ax, (6.5, 4.4), (5.0, 2.5), color=RED)

# --- future state ---
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.set_facecolor(SURFACE)
ax.set_title("FUTURE STATE — verification gate added", fontsize=11.5, color=GOOD, loc="left", weight="bold", pad=10)
steps_fs = [
    (0.1, 4.4, 1.9, 1.1, "Account placed\nwith collector"),
    (2.35, 4.05, 2.3, 1.75, "NEW: automated\nidentity + balance\nverification match"),
    (4.95, 4.4, 2.1, 1.1, "Verified →\ncontact consumer"),
    (7.35, 4.4, 2.35, 1.1, "Unverified → return\nto creditor, no contact"),
]
for x, y, w, h, t in steps_fs:
    fc = "#e4f4ec" if "NEW" in t else "#ffffff"
    ec = GOOD if "NEW" in t else GRID
    tc = "#0a5c2e" if "NEW" in t else INK
    box(ax, (x, y), w, h, t, fc, ec, tc=tc, fs=8.4, weight="bold" if "NEW" in t else "normal")
arrow(ax, (2.0, 4.4+0.55), (2.35, 4.4+0.55))
arrow(ax, (4.65, 4.85), (4.95, 4.85), color=GOOD)
arrow(ax, (4.65, 4.35), (7.35, 4.85), color=MUTED)
box(ax, (2.6, 1.4), 5.0, 1.1, "Target: cut “not owed” complaint\nvolume and its handling cost", "#e4f4ec", GOOD, tc="#0a5c2e", fs=8.8, weight="bold")
arrow(ax, (5.9, 4.05), (5.1, 2.5), color=GOOD)

plt.tight_layout()
plt.savefig("process_map_current_vs_future.png", facecolor=SURFACE)
plt.close()

print("Charts and process map saved.")
