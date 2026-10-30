"""
Builds RouteVoice's self-contained HTML dashboard: KPI cards, a 4-panel
Plotly view (regional containment lift, the confidence-vs-containment
regression, the ASR outage window, and confidence by region), and the
agent's reports for Appalachian, the ASR outage, and the benign deploy
rendered as verdict-coded cards.
"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from voice_metrics import containment_summary, containment_shift_test
from call_anomaly import hourly_confidence_series, point_anomaly_detector, find_anomaly_windows
from regression_analysis import fit_containment_model, standardized_effect_summary
from agent import review_regional_deploy, review_asr_outage, review_benign_deploy
from llm_client import get_client

calls = pd.read_csv("../data/calls.csv", parse_dates=["timestamp"])
client = get_client(cache_file="../cached_responses.json")

regions = ["Midwest", "Gulf_Coast", "Northeast", "Pacific_Northwest", "Appalachian"]
national = containment_shift_test(calls, region=None)
regional_results = pd.DataFrame([vars(containment_shift_test(calls, r)) for r in regions], index=regions)

model = fit_containment_model(calls)
effects = standardized_effect_summary(model).drop(index="const")

hourly = hourly_confidence_series(calls)
anomalies = point_anomaly_detector(hourly)
outage_windows = find_anomaly_windows(anomalies)

conf_by_region = containment_summary(calls, groupby=["driver_region", "model_version"])[["avg_confidence"]].unstack()
conf_by_region.columns = ["v1", "v2"]

regional_reports = {r: review_regional_deploy(calls, r, client) for r in regions}
asr_report = review_asr_outage(outage_windows.iloc[0].to_dict(), client)
benign_report = review_benign_deploy(calls, pd.Timestamp("2026-06-05"), window_days=3, client=client)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Containment Lift by Region (v1 → v2)", "Containment Drivers (Odds Ratio, Logistic Regression)",
    "ASR Outage: Hourly Transcription Confidence", "Transcription Confidence by Region (v1 → v2)",
))

colors = ["#27AE60" if lp > 0 else "#C0392B" for lp in regional_results.lift_pp]
fig.add_trace(go.Bar(x=regional_results.lift_pp, y=regional_results.index, orientation="h", marker_color=colors, showlegend=False), row=1, col=1)
fig.add_vline(x=national.lift_pp, line_dash="dash", line_color="#2E86AB", row=1, col=1)

fig.add_trace(go.Bar(x=effects.index, y=effects.odds_ratio, marker_color="#2E86AB", showlegend=False), row=1, col=2)
fig.add_hline(y=1, line_color="black", line_width=1, row=1, col=2)

window = hourly.loc["2026-05-12":"2026-05-14"]
fig.add_trace(go.Scatter(x=window.index, y=window.values, mode="lines", line=dict(color="#2E86AB", width=1.5), showlegend=False), row=2, col=1)
w = outage_windows.iloc[0]
fig.add_vrect(x0=w.start, x1=w.end, fillcolor="#C0392B", opacity=0.2, line_width=0, row=2, col=1)

for version, color in [("v1", "#95A5A6"), ("v2", "#2E86AB")]:
    fig.add_trace(go.Bar(x=conf_by_region.index, y=conf_by_region[version], name=version, marker_color=color), row=2, col=2)

fig.update_xaxes(title_text="Containment lift (pp)", row=1, col=1)
fig.update_yaxes(title_text="Odds ratio", row=1, col=2)
fig.update_yaxes(title_text="Confidence", row=2, col=1)
fig.update_yaxes(title_text="Avg confidence", row=2, col=2)
fig.update_layout(
    height=780, barmode="group",
    font=dict(family="Inter, Segoe UI, Arial", size=11.5),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=60, l=60, r=30, b=50),
    legend=dict(orientation="h", y=1.08),
)
html_chart = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})


def card(title, verdict, narrative, color):
    lines = narrative.strip().split("\n\n")
    body = "".join(f"<p>{l.replace(chr(10), '<br>')}</p>" for l in lines[:3])
    return f"""<div class="card">
      <span class="badge" style="background:{color}">{verdict.replace('_', ' ').title()}</span>
      <h3>{title}</h3>
      {body}
    </div>"""


cards_html = (
    card("Appalachian — model deploy", regional_reports["Appalachian"].verdict, regional_reports["Appalachian"].narrative, "#C0392B") +
    card("Midwest — model deploy (control)", regional_reports["Midwest"].verdict, regional_reports["Midwest"].narrative, "#27AE60") +
    card("ASR provider outage", asr_report.verdict, asr_report.narrative, "#E8A33D") +
    card("Greeting-script deploy", benign_report.verdict, benign_report.narrative, "#27AE60")
)

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RouteVoice — Voice AI Call Quality Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#FAFAFA; color:#1a1a1a; }}
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .sub {{ color:#666; font-size:14px; margin-bottom: 18px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:16px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:210px; }}
  .kpi .val {{ font-size:24px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.03em; margin-top:2px; }}
  .panel {{ background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:8px 16px; margin-bottom:20px; }}
  .cards {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin: 20px 0; }}
  .card {{ background:white; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:16px 18px; font-size:13px; line-height:1.6; }}
  .card h3 {{ margin:0 0 8px; font-size:14px; }}
  .card p {{ margin: 6px 0; }}
  .card .badge {{ display:inline-block; color:white; font-size:11px; font-weight:700; padding:2px 8px; border-radius:8px; margin-bottom:8px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
  @media (max-width: 900px) {{ .cards {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>📞 RouteVoice — Logistics Voice AI Agent Call Quality &amp; Containment Analytics</h1>
  <div class="sub">A voice-model release looks like a clean nationwide win — until you check region-by-region and find a regional dialect coverage gap hiding inside it.</div>
  <div class="kpis">
    <div class="kpi"><div class="val">+4.94pp</div><div class="lbl">Nationwide Containment Lift (looks clean)</div></div>
    <div class="kpi"><div class="val">−6.75pp</div><div class="lbl">Appalachian Containment Regression (p=3.6×10⁻⁶)</div></div>
    <div class="kpi"><div class="val">17.7x</div><div class="lbl">Odds Ratio: Transcription Confidence → Containment</div></div>
    <div class="kpi"><div class="val">14.1σ</div><div class="lbl">Peak ASR Outage Anomaly Z-Score</div></div>
  </div>
  <div class="panel">{html_chart}</div>
  <h2 style="font-size:16px;">Agent Reports</h2>
  <div class="cards">{cards_html}</div>
  <footer>Synthetic call-level telemetry, seeded and reproducible (see README) · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
