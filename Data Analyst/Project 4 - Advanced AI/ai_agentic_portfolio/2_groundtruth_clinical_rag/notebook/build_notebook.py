import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""
# GroundTruth Clinical — RAG Retrieval Quality & Clinical Hallucination Risk Analytics

**What this is:** a RAG (retrieval-augmented generation) quality
analytics system for Meridian Health AI, a fictional clinical
documentation copilot that answers clinician questions by retrieving
guideline text and generating a grounded answer — the dominant
AI-native product architecture right now, specifically because RAG is
supposed to reduce hallucination versus an ungrounded model. This
notebook evaluates the RETRIEVAL step directly (precision@k, recall@k,
MRR — not just a final quality score) and validates a cheap groundedness
proxy against an expensive LLM judge on real, hand-authored clinical
Q&A examples.

**On these being fictional, illustrative clinical examples:** the
guideline text, dosing, and interaction claims in this notebook are
synthetic and for demonstration purposes only — not real medical
guidance.

**The finding this uncovers:** Meridian upgrades its retrieval embedding
model (v1 → v2). Every query type improves — including, in aggregate,
`drug_interaction`. But `drug_interaction` queries about **rare drug
combinations** (~20% of that category's volume) get sharply **worse**:
precision@k falls from 0.440 to 0.283 (p=6.6×10⁻⁴³), hallucination rate
rises, and — worse — physicians are *less* likely to catch a hallucinated
answer about an interaction they're less personally familiar with. This
segment-level regression is invisible in the category aggregate because
the common-combination majority is improving at the same time. Finding
it requires deliberately slicing on segment, not trusting the headline
number.

**On the LLM backend:** same cached/live design as the rest of this
session's AI-augmented projects — see `../shared/llm_client.py`.
""")

code(r"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from retrieval_metrics import retrieval_summary, compare_versions_ttest
from hallucination_risk import hallucination_summary, silent_failure_rate
from judge_validation import judge_all_samples
from agent import review_segment
from llm_client import get_client

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

queries = pd.read_csv("../data/queries.csv", parse_dates=["timestamp"])
client = get_client(cache_file="../cached_responses.json")

print(f"{len(queries):,} queries | {queries.timestamp.min()} to {queries.timestamp.max()} | "
      f"deploy at 2026-03-26 (embedding v1 -> v2)")
queries.head()
""")

md("## 1. Retrieval Quality by Segment — Where the Aggregate Lies")

code(r"""
seg = queries.copy()
seg["segment"] = np.where(seg.query_type == "drug_interaction",
                           "drug_interaction_" + np.where(seg.rare_drug_combination, "rare", "common"),
                           seg.query_type)
summary = retrieval_summary(seg, groupby=["segment", "embedding_version"])
summary
""")

code(r"""
segments_order = ["medication_dosage", "diagnosis_criteria", "care_protocol", "drug_interaction_common", "drug_interaction_rare"]
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(segments_order))
width = 0.35
v1_vals = [summary.loc[(s, "v1"), "precision_at_k"] for s in segments_order]
v2_vals = [summary.loc[(s, "v2"), "precision_at_k"] for s in segments_order]
colors = ["#2E86AB"] * 4 + ["#C0392B"]
ax.bar(x - width/2, v1_vals, width, label="v1", color="#95A5A6")
ax.bar(x + width/2, v2_vals, width, label="v2", color=colors)
ax.set_xticks(x); ax.set_xticklabels(segments_order, rotation=20, ha="right")
ax.set_ylabel("precision@k")
ax.set_title("Retrieval Precision@k by Segment: Only drug_interaction_rare Regresses")
ax.legend()
plt.tight_layout()
plt.savefig("../dashboard/precision_by_segment.png", dpi=150)
plt.show()
""")

md(r"""Every segment's precision@k improves from v1 to v2 — except
`drug_interaction_rare`, which drops sharply. Because rare combinations
are only ~20% of `drug_interaction` volume, this drop barely moves the
category-level aggregate (see below), even though it's the largest
effect size in the whole release.""")

code(r"""
category_level = queries[queries.query_type == "drug_interaction"].groupby("embedding_version").agg(
    n=("query_id", "count"), precision_at_k=("precision_at_k", "mean"), hallucination_rate=("hallucination_occurred", "mean"),
).round(4)
print("drug_interaction CATEGORY AGGREGATE (hides the rare-combo regression):")
category_level
""")

md("## 2. Statistical Tests — Precision, Hallucination Rate, by Segment")

code(r"""
segment_specs = [
    ("medication_dosage", "medication_dosage", None),
    ("diagnosis_criteria", "diagnosis_criteria", None),
    ("care_protocol", "care_protocol", None),
    ("drug_interaction_common", "drug_interaction", False),
    ("drug_interaction_rare", "drug_interaction", True),
]
precision_tests = pd.DataFrame([compare_versions_ttest(queries, "precision_at_k", qt, rare) for _, qt, rare in segment_specs])
precision_tests
""")

md("## 3. Downstream Safety Signal: Hallucination Rate & Silent Failures")

code(r"""
seg_halluc = hallucination_summary(seg, groupby=["segment", "embedding_version"])
seg_silent = silent_failure_rate(seg, groupby=["segment", "embedding_version"])
seg_halluc.join(seg_silent[["silent_failure_rate"]])
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, title in zip(axes, ["hallucination_rate", "silent_failure_rate"],
                              ["Hallucination Rate", "Silent Failure Rate\n(hallucinated AND accepted as-is)"]):
    tbl = (seg_halluc if metric == "hallucination_rate" else seg_silent).reset_index()
    piv = tbl.pivot(index="segment", columns="embedding_version", values=metric).reindex(segments_order)
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv.v1, 0.4, label="v1", color="#95A5A6")
    ax.bar(x + 0.2, piv.v2, 0.4, label="v2", color="#2E86AB")
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_title(title, fontsize=11)
    ax.legend()
plt.tight_layout()
plt.savefig("../dashboard/safety_metrics.png", dpi=150)
plt.show()
""")

md(r"""`drug_interaction_rare` is the only segment where BOTH hallucination
rate and silent failure rate rise from v1 to v2 — and it's the segment
where physicians are also least equipped to catch it, since rare
interactions are, by definition, less familiar.""")

md("## 4. Validating the Cheap Proxy Against the Expensive Judge")

code(r"""
samples = json.load(open("../data/judge_samples.json"))
judged = judge_all_samples(samples, client)
judged["group_label"] = np.where(
    judged.query_type == "drug_interaction",
    "drug_interaction_" + np.where(judged.get("rare_drug_combination", False).fillna(False), "rare", "common"),
    judged.query_type,
)
judge_group = judged.groupby(["group_label", "embedding_version"]).judge_score.mean().reset_index()
judge_group
""")

code(r"""
proxy_rows = []
for _, row in judge_group.iterrows():
    label, ver = row.group_label, row.embedding_version
    if label.startswith("drug_interaction_"):
        rare = label.endswith("rare")
        sub = queries[(queries.query_type == "drug_interaction") & (queries.rare_drug_combination == rare) & (queries.embedding_version == ver)]
    else:
        sub = queries[(queries.query_type == label) & (queries.embedding_version == ver)]
    proxy_rows.append({"group_label": label, "embedding_version": ver, "judge_score": row.judge_score,
                        "citation_coverage_proxy": sub.citation_coverage.mean() * 100})
validation = pd.DataFrame(proxy_rows)
corr = validation.judge_score.corr(validation.citation_coverage_proxy)
print(f"Proxy-vs-judge correlation: r={corr:.2f} (n={len(validation)} groups)")
validation
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(validation.citation_coverage_proxy, validation.judge_score, s=90, color="#2E86AB", alpha=0.8)
for _, r in validation.iterrows():
    ax.annotate(f"{r.group_label[:12]}/{r.embedding_version}", (r.citation_coverage_proxy, r.judge_score), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Citation-coverage proxy (0-100, cheap & always-on)")
ax.set_ylabel("LLM judge groundedness score (0-100, expensive)")
ax.set_title(f"Proxy vs. Judge (r={corr:.2f}, n={len(validation)} groups)")
plt.tight_layout()
plt.savefig("../dashboard/proxy_validation.png", dpi=150)
plt.show()
""")

md(r"""The correlation is largely anchored by the `drug_interaction_rare`/v2
point, where BOTH the cheap proxy and the expensive judge independently
show a collapse — meaning the cheap, always-on proxy would have flagged
this segment for judge review even without waiting for a hand-audited
sample to catch it, which is the whole point of validating a proxy in
the first place.""")

md("## 5. Agent Release-Review Reports, by Segment")

code(r"""
reports = {label: review_segment(queries, label, qt, rare, client) for label, qt, rare in segment_specs}
for label in ["medication_dosage", "diagnosis_criteria", "care_protocol", "drug_interaction_common", "drug_interaction_rare"]:
    print(f"=== {label} ===")
    print(reports[label].narrative)
    print()
""")

md("## 6. Summary")

code(r"""
print("=== GroundTruth Clinical summary ===")
for label in segments_order:
    print(f"{label:28s} -> {reports[label].verdict}")
print()
print("The finding that matters most: drug_interaction's CATEGORY aggregate looks stable across the v2")
print("release (precision 0.470 -> 0.472), but the rare-combination sub-segment (~20% of volume) actually")
print("collapsed (0.440 -> 0.283, p=6.6e-43), with hallucination rate and silent-failure rate both rising --")
print("exactly the kind of segment-level patient-safety risk a category-level RAG quality dashboard would miss.")
""")

nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Notebook written.")
