"""
CASE 08 — AI-Augmented Qualitative Theme Coding at Scale
Deterministic analysis stage: consumes llm_theme_codes.json (the actual
open-coding pass performed on 120 real Coursera reviews) plus the source
CSV for context, and produces the frequency table, the theme x rating-band
cross-tab, and the two report charts.
"""
import json
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# ---------------------------------------------------------------
# Context: full platform rating skew (for the honesty section)
# ---------------------------------------------------------------
full = pd.read_csv("../data/coursera_reviews.csv")
full_dist = full["Label"].value_counts().sort_index().to_dict()
full_n = len(full)

# ---------------------------------------------------------------
# Load the coded sample
# ---------------------------------------------------------------
coded = json.load(open("llm_theme_codes.json"))
reviews = coded["reviews"]
taxonomy = coded["taxonomy"]
n_sample = len(reviews)


def band(stars):
    if stars <= 2:
        return "1-2 star (unhappy)"
    elif stars == 3:
        return "3 star (mixed)"
    else:
        return "4-5 star (happy)"


band_counts = Counter(band(r["stars"]) for r in reviews)

# ---------------------------------------------------------------
# Theme frequency (count of REVIEWS carrying each theme, across the
# 120-review sample; a review can carry up to 2 themes)
# ---------------------------------------------------------------
theme_counts = Counter()
for r in reviews:
    for t in r["themes"]:
        theme_counts[t] += 1

theme_freq_table = (
    pd.Series(theme_counts)
    .rename("review_count")
    .to_frame()
    .assign(theme_label=lambda d: [taxonomy[i] for i in d.index],
            pct_of_sample=lambda d: (d["review_count"] / n_sample * 100).round(1))
    .sort_values("review_count", ascending=False)
    .reset_index().rename(columns={"index": "theme_code"})
    [["theme_code", "theme_label", "review_count", "pct_of_sample"]]
)

# ---------------------------------------------------------------
# Cross-tab: theme x rating band
# ---------------------------------------------------------------
crosstab = defaultdict(lambda: defaultdict(int))
for r in reviews:
    b = band(r["stars"])
    for t in r["themes"]:
        crosstab[t][b] += 1

band_order = ["1-2 star (unhappy)", "3 star (mixed)", "4-5 star (happy)"]
crosstab_rows = []
for t in theme_counts:
    row = {"theme_code": t, "theme_label": taxonomy[t]}
    for b in band_order:
        row[b] = crosstab[t].get(b, 0)
    crosstab_rows.append(row)
crosstab_df = pd.DataFrame(crosstab_rows).set_index("theme_code")
crosstab_df = crosstab_df.loc[theme_freq_table.set_index("theme_code").index]

# ---------------------------------------------------------------
# Save results JSON
# ---------------------------------------------------------------
top_theme_code = theme_freq_table.iloc[0]["theme_code"]
top_theme_label = theme_freq_table.iloc[0]["theme_label"]
top_theme_n = int(theme_freq_table.iloc[0]["review_count"])
unhappy_n = band_counts["1-2 star (unhappy)"]
top_theme_unhappy_n = int(crosstab_df.loc[top_theme_code, "1-2 star (unhappy)"])
top_theme_unhappy_share = round(top_theme_unhappy_n / unhappy_n * 100, 1)

results = {
    "full_dataset_n": int(full_n),
    "full_dataset_rating_distribution": {str(k): int(v) for k, v in full_dist.items()},
    "sample_n": n_sample,
    "sample_rating_band_counts": dict(band_counts),
    "taxonomy": taxonomy,
    "theme_frequency": theme_freq_table.to_dict(orient="records"),
    "theme_by_rating_band_crosstab": crosstab_df.reset_index().to_dict(orient="records"),
    "top_theme_code": top_theme_code,
    "top_theme_label": top_theme_label,
    "top_theme_review_count": top_theme_n,
    "top_theme_pct_of_sample": round(top_theme_n / n_sample * 100, 1),
    "top_theme_share_of_unhappy_reviews": top_theme_unhappy_share,
}
with open("theme_mining_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps({k: v for k, v in results.items() if k not in
                   ("theme_by_rating_band_crosstab", "full_dataset_rating_distribution")}, indent=2))

# ---------------------------------------------------------------
# CHART 1 — theme frequency (horizontal bar, top theme highlighted)
# ---------------------------------------------------------------
s = theme_freq_table.set_index("theme_label")["review_count"].sort_values(ascending=True)
colors_bar = [RED if i == len(s) - 1 else MUTED for i in range(len(s))]
fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.barh(s.index, s.values, color=colors_bar, height=0.58, zorder=3)
for b, v in zip(bars, s.values):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v}", va="center", fontsize=9.5, color=INK)
ax.set_title(
    f"\"{top_theme_label}\" is the single most common theme\n"
    f"(n={n_sample} reviews, LLM open-coded, 1-2 themes per review)",
    fontsize=11, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_xlim(0, s.values.max() * 1.16)
plt.tight_layout()
plt.savefig("chart_theme_frequency.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — theme mix, unhappy (1-2*) vs happy (4-5*) band
# ---------------------------------------------------------------
plot_df = crosstab_df[["1-2 star (unhappy)", "4-5 star (happy)"]].copy()
plot_df.index = [taxonomy[c] for c in plot_df.index]
# order by unhappy count descending for readability
plot_df = plot_df.sort_values("1-2 star (unhappy)", ascending=True)

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
y = range(len(plot_df))
h = 0.36
b1 = ax.barh([p + h / 2 for p in y], plot_df["1-2 star (unhappy)"], height=h, color=RED, label="1-2 star reviews", zorder=3)
b2 = ax.barh([p - h / 2 for p in y], plot_df["4-5 star (happy)"], height=h, color=AQUA, label="4-5 star reviews", zorder=3)
for bars in (b1, b2):
    for b in bars:
        w = b.get_width()
        if w > 0:
            ax.text(w + 0.3, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center", fontsize=9, color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels(plot_df.index, fontsize=9.5, color=INK2)
ax.set_title("The theme mix flips between unhappy and happy reviewers",
             fontsize=11.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
max_val = max(plot_df["1-2 star (unhappy)"].max(), plot_df["4-5 star (happy)"].max())
ax.set_xlim(0, max_val * 1.2)
ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK2)
plt.tight_layout()
plt.savefig("chart_theme_by_rating.png", facecolor=SURFACE)
plt.close()

print("\nCharts and theme_mining_results.json saved.")
