"""
CASE 06 — Campaign Channel Effectiveness: A Statistical Test (Banking)
Real UCI Bank Marketing dataset (41,188 telemarketing contacts, Portuguese
bank, direct term-deposit campaign). Chi-square + two-proportion z-test on
contact channel, with a power analysis for a proposed real A/B test.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

plt.rcParams["font.family"] = "DejaVu Sans"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; RED="#e34948"; GOOD="#0ca30c"
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; SURFACE="#fcfcfb"

df = pd.read_csv("../data/bank_marketing.csv")
df["y_bin"] = (df["y"] == "yes").astype(int)

# ---------------------------------------------------------------
# Primary test: contact channel (cellular vs telephone)
# ---------------------------------------------------------------
ct = pd.crosstab(df["contact"], df["y"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

cell = df[df["contact"] == "cellular"]
tel = df[df["contact"] == "telephone"]
n_cell, x_cell = len(cell), cell["y_bin"].sum()
n_tel, x_tel = len(tel), tel["y_bin"].sum()
p_cell, p_tel = x_cell / n_cell, x_tel / n_tel

z_stat, p_z = proportions_ztest([x_cell, x_tel], [n_cell, n_tel])
ci_cell = proportion_confint(x_cell, n_cell, alpha=0.05, method="wilson")
ci_tel = proportion_confint(x_tel, n_tel, alpha=0.05, method="wilson")
effect_size_h = proportion_effectsize(p_cell, p_tel)

lift_pct = 100 * (p_cell - p_tel) / p_tel

results = {
    "n_total": int(len(df)),
    "overall_conversion_pct": round(100 * df["y_bin"].mean(), 2),
    "n_cellular": int(n_cell), "x_cellular": int(x_cell), "p_cellular": round(float(p_cell), 4),
    "n_telephone": int(n_tel), "x_telephone": int(x_tel), "p_telephone": round(float(p_tel), 4),
    "ci_cellular": [round(float(c), 4) for c in ci_cell],
    "ci_telephone": [round(float(c), 4) for c in ci_tel],
    "chi2": round(float(chi2), 2), "chi2_p_value": float(p_chi2),
    "z_stat": round(float(z_stat), 2), "z_p_value": float(p_z),
    "effect_size_cohens_h": round(float(effect_size_h), 3),
    "lift_pct": round(float(lift_pct), 1),
}

# ---------------------------------------------------------------
# Secondary test: prior campaign success (poutcome)
# ---------------------------------------------------------------
prev_success = df[df["poutcome"] == "success"]
prev_other = df[df["poutcome"] != "success"]
p_prev_succ = prev_success["y_bin"].mean()
p_prev_other = prev_other["y_bin"].mean()
z2, p_z2 = proportions_ztest([prev_success["y_bin"].sum(), prev_other["y_bin"].sum()],
                              [len(prev_success), len(prev_other)])
results["secondary_test"] = {
    "n_prev_success": int(len(prev_success)), "p_prev_success": round(float(p_prev_succ), 4),
    "n_prev_other": int(len(prev_other)), "p_prev_other": round(float(p_prev_other), 4),
    "z_stat": round(float(z2), 2), "p_value": float(p_z2),
}

# ---------------------------------------------------------------
# Power analysis for a PROPOSED future randomized A/B test
# (channel choice here was NOT randomized — this is the honest fix)
# ---------------------------------------------------------------
power_analysis = NormalIndPower()
baseline_p = p_tel  # conservative baseline: the lower-converting channel
mde_scenarios = [0.01, 0.02, 0.03]  # minimum detectable lift, absolute percentage points
sample_size_rows = []
for mde in mde_scenarios:
    p2 = baseline_p + mde
    h = proportion_effectsize(p2, baseline_p)
    n_needed = power_analysis.solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")
    sample_size_rows.append({"mde_pp": mde, "n_per_arm": int(np.ceil(n_needed))})
results["power_analysis"] = sample_size_rows

with open("stats_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------
# CHART 1 — Conversion rate by channel with 95% CI
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
cats = ["Telephone", "Cellular"]
vals = [p_tel*100, p_cell*100]
errs_low = [ (p_tel - ci_tel[0])*100, (p_cell - ci_cell[0])*100 ]
errs_high = [ (ci_tel[1] - p_tel)*100, (ci_cell[1] - p_cell)*100 ]
colors = [MUTED, GOOD]
bars = ax.bar(cats, vals, color=colors, width=0.5, zorder=3,
               yerr=[errs_low, errs_high], capsize=6, ecolor=INK2, error_kw={"linewidth":1.3})
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v + 1.3, f"{v:.1f}%", ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_ylabel("Term-deposit conversion rate", fontsize=10, color=INK2)
ax.set_title(f"Cellular converts {results['lift_pct']:.0f}% better (95% CI, non-overlapping)", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=10.5, colors=INK2)
ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
plt.tight_layout()
plt.savefig("chart_conversion_by_channel.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — Required sample size vs. minimum detectable effect
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
mdes = [f"+{int(r['mde_pp']*100)}pp" for r in sample_size_rows]
ns = [r["n_per_arm"] for r in sample_size_rows]
bars = ax.bar(mdes, ns, color=BLUE, width=0.45, zorder=3)
for b, v in zip(bars, ns):
    ax.text(b.get_x()+b.get_width()/2, v + max(ns)*0.02, f"{v:,}", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.set_xlabel("Minimum detectable lift over baseline", fontsize=9.5, color=INK2)
ax.set_ylabel("Required sample size per arm", fontsize=9.5, color=INK2)
ax.set_title("Smaller effects need much bigger test samples", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=10, colors=INK2)
ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
plt.tight_layout()
plt.savefig("chart_power_analysis.png", facecolor=SURFACE)
plt.close()

print("\nCharts saved.")
