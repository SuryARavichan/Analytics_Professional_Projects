"""
Builds GroundTruth Clinical's self-contained HTML dashboard: KPI cards,
a 4-panel Plotly view (precision@k by segment, hallucination/silent-
failure rates, the proxy-vs-judge validation scatter), and the two real
hallucinated clinical examples rendered as patient-safety evidence cards
alongside the five segment release-review reports.
"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from retrieval_metrics import retrieval_summary, compare_versions_ttest
from hallucination_risk import hallucination_summary, silent_failure_rate
from judge_validation import judge_all_samples
from agent import review_segment
from llm_client import get_client

queries = pd.read_csv("../data/queries.csv", parse_dates=["timestamp"])
client = get_client(cache_file="../cached_responses.json")

seg = queries.copy()
seg["segment"] = np.where(seg.query_type == "drug_interaction",
                           "drug_interaction_" + np.where(seg.rare_drug_combination, "rare", "common"),
                           seg.query_type)
segments_order = ["medication_dosage", "diagnosis_criteria", "care_protocol", "drug_interaction_common", "drug_interaction_rare"]
summary = retrieval_summary(seg, groupby=["segment", "embedding_version"])
seg_halluc = hallucination_summary(seg, groupby=["segment", "embedding_version"])
seg_silent = silent_failure_rate(seg, groupby=["segment", "embedding_version"])

segment_specs = [
    ("medication_dosage", "medication_dosage", None), ("diagnosis_criteria", "diagnosis_criteria", None),
    ("care_protocol", "care_protocol", None), ("drug_interaction_common", "drug_interaction", False),
    ("drug_interaction_rare", "drug_interaction", True),
]
reports = {label: review_segment(queries, label, qt, rare, client) for label, qt, rare in segment_specs}

samples = json.load(open("../data/judge_samples.json"))
judged = judge_all_samples(samples, client)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Precision@k by Segment (v1 → v2)", "Hallucination Rate by Segment (v1 → v2)",
    "Silent Failure Rate by Segment (v1 → v2)", "Proxy vs. Judge Groundedness Score",
))

v1_prec = [summary.loc[(s, "v1"), "precision_at_k"] for s in segments_order]
v2_prec = [summary.loc[(s, "v2"), "precision_at_k"] for s in segments_order]
fig.add_trace(go.Bar(x=segments_order, y=v1_prec, name="v1", marker_color="#95A5A6"), row=1, col=1)
fig.add_trace(go.Bar(x=segments_order, y=v2_prec, name="v2", marker_color="#2E86AB", showlegend=False), row=1, col=1)

v1_hal = [seg_halluc.loc[(s, "v1"), "hallucination_rate"] for s in segments_order]
v2_hal = [seg_halluc.loc[(s, "v2"), "hallucination_rate"] for s in segments_order]
fig.add_trace(go.Bar(x=segments_order, y=v1_hal, name="v1 halluc", marker_color="#95A5A6", showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=segments_order, y=v2_hal, name="v2 halluc", marker_color="#C0392B", showlegend=False), row=1, col=2)

v1_sf = [seg_silent.loc[(s, "v1"), "silent_failure_rate"] for s in segments_order]
v2_sf = [seg_silent.loc[(s, "v2"), "silent_failure_rate"] for s in segments_order]
fig.add_trace(go.Bar(x=segments_order, y=v1_sf, name="v1 silent fail", marker_color="#95A5A6", showlegend=False), row=2, col=1)
fig.add_trace(go.Bar(x=segments_order, y=v2_sf, name="v2 silent fail", marker_color="#C0392B", showlegend=False), row=2, col=1)

judged["group_label"] = np.where(
    judged.query_type == "drug_interaction",
    "drug_interaction_" + np.where(judged.get("rare_drug_combination", False).fillna(False), "rare", "common"),
    judged.query_type,
)
judge_group = judged.groupby(["group_label", "embedding_version"]).judge_score.mean().reset_index()
proxy_rows = []
for _, row in judge_group.iterrows():
    label, ver = row.group_label, row.embedding_version
    if label.startswith("drug_interaction_"):
        rare = label.endswith("rare")
        sub = queries[(queries.query_type == "drug_interaction") & (queries.rare_drug_combination == rare) & (queries.embedding_version == ver)]
    else:
        sub = queries[(queries.query_type == label) & (queries.embedding_version == ver)]
    proxy_rows.append({"group_label": label, "embedding_version": ver, "judge_score": row.judge_score,
                        "proxy": sub.citation_coverage.mean() * 100})
validation = pd.DataFrame(proxy_rows)
corr = validation.judge_score.corr(validation.proxy)
fig.add_trace(go.Scatter(x=validation.proxy, y=validation.judge_score, mode="markers",
                          marker=dict(size=12, color="#2E86AB"), showlegend=False), row=2, col=2)

fig.update_yaxes(title_text="precision@k", row=1, col=1)
fig.update_yaxes(title_text="Hallucination rate", row=1, col=2)
fig.update_yaxes(title_text="Silent failure rate", row=2, col=1)
fig.update_xaxes(title_text="Citation-coverage proxy", row=2, col=2)
fig.update_yaxes(title_text="Judge groundedness score", row=2, col=2)
for r, c in [(1, 1), (1, 2), (2, 1)]:
    fig.update_xaxes(tickangle=20, row=r, col=c)
fig.update_layout(
    height=780, barmode="group",
    font=dict(family="Inter, Segoe UI, Arial", size=11),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=60, l=60, r=30, b=80),
    legend=dict(orientation="h", y=1.08),
)
html_chart = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})

hallu1 = judged.loc[judged.sample_id == "di_rare_v2_a"].iloc[0]
hallu2 = judged.loc[judged.sample_id == "di_rare_v2_b"].iloc[0]


def evidence_card(row):
    rationale = row.rationale_text.split("\n", 1)[1].replace("RATIONALE: ", "") if "\n" in row.rationale_text else row.rationale_text
    return f"""<div class="card">
      <span class="badge">SCORE: {row.judge_score}/100</span>
      <h3>{row.sample_id} — {row.query_type} (rare combo) / {row.embedding_version}</h3>
      <p><b>Q:</b> {row['query']}</p>
      <p><b>Retrieved context:</b> {row.retrieved_context}</p>
      <pre>{row.generated_answer}</pre>
      <div class="rationale">{rationale}</div>
    </div>"""


def verdict_card(label, verdict, narrative):
    color = "#C0392B" if verdict == "PATIENT_SAFETY_RISK" else "#27AE60"
    lines = narrative.strip().split("\n\n")
    body = "".join(f"<p>{l.replace(chr(10), '<br>')}</p>" for l in lines[:3])
    return f"""<div class="card">
      <span class="badge" style="background:{color}">{verdict.replace('_', ' ').title()}</span>
      <h3>{label.replace('_', ' ').title()}</h3>
      {body}
    </div>"""


evidence_html = evidence_card(hallu1) + evidence_card(hallu2)
verdict_html = "".join(verdict_card(label, reports[label].verdict, reports[label].narrative) for label, _, _ in segment_specs)

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GroundTruth Clinical — RAG Quality Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#FAFAFA; color:#1a1a1a; }}
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .sub {{ color:#666; font-size:14px; margin-bottom: 18px; }}
  .disclaimer {{ font-size:11px; color:#999; margin-bottom:14px; font-style:italic; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:16px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:210px; }}
  .kpi .val {{ font-size:24px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.03em; margin-top:2px; }}
  .panel {{ background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:8px 16px; margin-bottom:20px; }}
  .cards {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin: 20px 0; }}
  .card {{ background:white; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:16px 18px; font-size:13px; line-height:1.6; }}
  .card h3 {{ margin:0 0 8px; font-size:14px; }}
  .card p {{ margin: 6px 0; }}
  .card .badge {{ display:inline-block; background:#C0392B; color:white; font-size:11px; font-weight:700; padding:2px 8px; border-radius:8px; margin-bottom:8px; }}
  .card pre {{ background:#1e1e2e; color:#cdd6f4; padding:10px 12px; border-radius:6px; font-size:12px; overflow-x:auto; white-space:pre-wrap; }}
  .card .rationale {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:8px 12px; border-radius:6px; margin-top:8px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
  @media (max-width: 900px) {{ .cards {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🏥 GroundTruth Clinical — RAG Retrieval Quality &amp; Hallucination Risk Analytics</h1>
  <div class="sub">Retrieval-quality (precision@k/recall@k/MRR) and clinical-groundedness analytics for a RAG-based clinical copilot — catching a rare-drug-combination hallucination risk hidden inside an improving category aggregate.</div>
  <div class="disclaimer">Synthetic, illustrative clinical content for demonstration only — not real medical guidance.</div>
  <div class="kpis">
    <div class="kpi"><div class="val">0.440 → 0.283</div><div class="lbl">Rare-Combo precision@k (p=6.6×10⁻⁴³)</div></div>
    <div class="kpi"><div class="val">+5.4pp</div><div class="lbl">Rare-Combo Hallucination Rate Increase</div></div>
    <div class="kpi"><div class="val">r={corr:.2f}</div><div class="lbl">Proxy-vs-Judge Correlation (n={len(validation)} groups)</div></div>
    <div class="kpi"><div class="val">4 / 5</div><div class="lbl">Segments Confirmed Improved</div></div>
  </div>
  <div class="panel">{html_chart}</div>
  <h2 style="font-size:16px;">The Regression, In the Model's Own Words</h2>
  <div class="cards">{evidence_html}</div>
  <h2 style="font-size:16px;">Segment Release-Review Verdicts</h2>
  <div class="cards">{verdict_html}</div>
  <footer>Synthetic, illustrative clinical query data, seeded and reproducible (see README) · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
