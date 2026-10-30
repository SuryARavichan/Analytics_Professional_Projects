"""
CASE 07 — Deterministic scoring & chart-building step.

This script does NOT re-run the LLM judgment (that's the manual/agentic step,
already captured in llm_judgments.json). It reads that file plus the real
EMSCAD ground-truth labels, recomputes every metric from scratch, and builds
the two report charts. Re-running this script always reproduces identical
numbers from the same llm_judgments.json.
"""
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; RED = "#e34948"; GOOD = "#0ca30c"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
judgments = json.load(open("llm_judgments.json"))
df_full = pd.read_csv("../data/fake_job_postings.csv")
true_labels = dict(zip(df_full["job_id"], df_full["fraudulent"]))

n_sample = len(judgments)
n_fraud_sample = sum(1 for j in judgments if true_labels[j["id"]] == 1)
n_legit_sample = n_sample - n_fraud_sample
population_base_rate = float(df_full["fraudulent"].mean())

# ---------------------------------------------------------------
# Confusion matrix + metrics
# ---------------------------------------------------------------
TP = FP = TN = FN = 0
for j in judgments:
    pred = bool(j["fraud_suspected"])
    actual = true_labels[j["id"]] == 1
    if pred and actual:
        TP += 1
    elif pred and not actual:
        FP += 1
    elif not pred and actual:
        FN += 1
    else:
        TN += 1

accuracy = (TP + TN) / n_sample
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

# ---------------------------------------------------------------
# Red-flag theme tally (from the LLM's own flagged-posting reasons)
# ---------------------------------------------------------------
flag_counter = Counter()
for j in judgments:
    if j["fraud_suspected"]:
        for flag in j["red_flags"]:
            flag_counter[flag] += 1

results = {
    "n_sample": n_sample,
    "n_fraud_sample": n_fraud_sample,
    "n_legit_sample": n_legit_sample,
    "population_base_rate": round(population_base_rate, 4),
    "population_n": int(len(df_full)),
    "confusion_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
    "accuracy": round(accuracy, 3),
    "precision": round(precision, 3),
    "recall": round(recall, 3),
    "f1": round(f1, 3),
    "n_flagged_by_llm": TP + FP,
    "red_flag_theme_counts": dict(flag_counter.most_common()),
}
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------
# CHART 1 — Confusion matrix
# ---------------------------------------------------------------
cm = [[TN, FP], [FN, TP]]
fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
im = ax.imshow(cm, cmap="Oranges", alpha=0.85)
labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
cm_max = max(max(row) for row in cm)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i][j]}\n{labels[i][j]}", ha="center", va="center", fontsize=11,
                color=INK if cm[i][j] < cm_max * 0.6 else "white", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["LLM says:\nLegitimate", "LLM says:\nFraud suspected"], fontsize=9.5, color=INK2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actually:\nLegitimate", "Actually:\nFraudulent"], fontsize=9.5, color=INK2)
ax.set_title(f"LLM fraud screen vs. confirmed EMSCAD labels\n(n={n_sample} stratified postings)", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("chart_confusion_matrix.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — Top red-flag themes (horizontal bar chart)
# ---------------------------------------------------------------
theme_labels = {
    "no verifiable company identity": "No verifiable company identity",
    "urgency / apply-immediately pressure language": "Urgency / apply-immediately pressure",
    "salary implausible for stated role/location": "Salary implausible for role",
    "vague/generic responsibilities": "Vague / generic responsibilities",
    "processing-fee or equipment-purchase language": "Fee / equipment-purchase language",
}
items = sorted(flag_counter.items(), key=lambda kv: kv[1])
labels_plot = [theme_labels.get(k, k) for k, v in items]
values_plot = [v for k, v in items]
bar_colors = [RED if i == len(items) - 1 else MUTED for i in range(len(items))]

fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.barh(labels_plot, values_plot, color=bar_colors, height=0.6, zorder=3)
for b, v in zip(bars, values_plot):
    ax.text(v + 0.2, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9.5, color=INK)
ax.set_title("Missing company identity is the single biggest tell",
             fontsize=13, color=INK, loc="left", pad=12, fontweight="bold")
ax.set_xlabel(f"Postings flagged citing this reason (of {TP + FP} flagged)", fontsize=9.5, color=INK2)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=10, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_xlim(0, max(values_plot) * 1.22)
plt.tight_layout()
plt.savefig("chart_red_flags.png", facecolor=SURFACE)
plt.close()

print("\nCharts saved: chart_confusion_matrix.png, chart_red_flags.png")
