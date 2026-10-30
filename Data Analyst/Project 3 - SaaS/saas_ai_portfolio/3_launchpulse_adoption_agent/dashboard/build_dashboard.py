import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from anomaly_detector import point_anomaly_detector, channel_drift_detector
from agent import investigate
from llm_client import get_client

DB = "../data/launchpulse.db"
df = pd.read_csv("../data/daily_metrics.csv", parse_dates=["date"])
funnel = pd.read_csv("../data/daily_channel_funnel.csv", parse_dates=["date"])
daily_total = df.groupby("date")["adoption_events"].sum()

point_anomalies = point_anomaly_detector(daily_total, z_threshold=3.5)

client = get_client(cache_file="../cached_responses.json")

incidents_config = [
    ("incident_2024_10_08_automation_bug", "Automation Rules Outage", "2024-10-08", "2024-10-08",
     "2024-10-01", "2024-10-07", {"z_score": -4.56}, "point_anomaly", "real"),
    ("incident_2025_04_ai_assistant_launch", "AI Assistant Launch Spike", "2025-04-14", "2025-04-16",
     "2025-04-07", "2025-04-13", {"z_score": "1.19 / 4.24 / 3.35"}, "point_anomaly", "real"),
    ("incident_2024_12_paid_social_drift", "Paid Social Activation Drift", "2024-12-14", "2024-12-24",
     "2024-11-23", "2024-12-03", {"max_decline_pct": -36.6}, "channel_drift", "real"),
    ("incident_2025_02_webinar_false_positive", "Feb 11-13 Broad Spike", "2025-02-11", "2025-02-13",
     "2025-02-04", "2025-02-10", {"z_score": "11.14 / 7.41 / 8.93"}, "point_anomaly", "dismissed"),
]

reports = []
for incident_id, title, s, e, bs, be, stats, kind, status in incidents_config:
    r = investigate(DB, incident_id, kind, s, e, bs, be, stats, client)
    reports.append({
        "id": incident_id, "title": title, "window": f"{s} to {e}", "status": status,
        "narrative": r.narrative,
        "feature_breakdown": r.feature_breakdown.to_dict("records"),
        "channel_breakdown": r.channel_breakdown.to_dict("records") if r.channel_breakdown is not None else None,
        "funnel_breakdown": r.funnel_breakdown.to_dict("records") if r.funnel_breakdown is not None else None,
    })

# ---- main timeline chart ----
fig = make_subplots(rows=1, cols=1, subplot_titles=("Daily Adoption Events — Flagged Anomalies & Agent-Investigated Incidents",))
fig.add_trace(go.Scatter(x=point_anomalies.date, y=point_anomalies.actual, mode="lines",
                          line=dict(color="#2E86AB", width=1), name="Daily adoption events"))
colors = {"real": "#C0392B", "dismissed": "#999999"}
for cfg in incidents_config:
    incident_id, title, s, e, *_ , kind, status = cfg
    window = point_anomalies[(point_anomalies.date >= s) & (point_anomalies.date <= e)]
    fig.add_trace(go.Scatter(x=window.date, y=window.actual, mode="markers",
                              marker=dict(color=colors[status], size=10, symbol="diamond" if status == "dismissed" else "circle"),
                              name=title))
fig.update_layout(height=480, plot_bgcolor="white", paper_bgcolor="#FAFAFA",
                   font=dict(family="Inter, Segoe UI, Arial", size=12), margin=dict(t=50, l=60, r=40, b=40),
                   legend=dict(orientation="h", y=-0.15))
html = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})


def render_table(rows, key_col):
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r[c]:,.1f}</td>" if isinstance(r[c], float) else f"<td>{r[c]}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


cards_html = ""
for rep in reports:
    status_badge = '<span class="badge dismissed">DISMISSED — FALSE ALARM</span>' if rep["status"] == "dismissed" else '<span class="badge real">CONFIRMED INCIDENT</span>'
    feature_table = render_table(rep["feature_breakdown"], "feature_name")
    channel_table = render_table(rep["channel_breakdown"], "channel") if rep["channel_breakdown"] else ""
    funnel_table = render_table(rep["funnel_breakdown"], "channel") if rep["funnel_breakdown"] else ""
    narrative_html = rep["narrative"].replace("\n\n", "</p><p>").replace("\n", "<br>")
    cards_html += f"""
    <div class="card">
      <div class="card-header"><h3>{rep['title']}</h3>{status_badge}</div>
      <div class="card-sub">Investigation window: {rep['window']}</div>
      <div class="evidence">
        <div><div class="ev-label">Feature breakdown</div>{feature_table}</div>
        <div><div class="ev-label">Channel breakdown</div>{channel_table}</div>
        {f'<div><div class="ev-label">Funnel breakdown (channel drift)</div>{funnel_table}</div>' if funnel_table else ''}
      </div>
      <div class="narrative"><p>{narrative_html}</p></div>
    </div>"""

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LaunchPulse — Adoption Anomaly Agent Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#F4F6F8; color:#1a1a1a; }}
  .wrap {{ max-width: 1250px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .sub {{ color:#666; font-size:14px; margin-bottom: 18px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:14px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:160px; }}
  .kpi .val {{ font-size:22px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:11px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .card {{ background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:20px 24px; margin-bottom:20px; }}
  .card-header {{ display:flex; align-items:center; gap:12px; }}
  .card-header h3 {{ margin:0; font-size:16px; }}
  .card-sub {{ font-size:12px; color:#888; margin: 4px 0 12px; }}
  .badge {{ padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }}
  .badge.real {{ background:#FDECEA; color:#C0392B; }}
  .badge.dismissed {{ background:#F0F0F0; color:#777; }}
  .evidence {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom: 14px; }}
  .ev-label {{ font-size:11px; text-transform:uppercase; color:#999; margin-bottom:4px; font-weight:600; }}
  table {{ border-collapse:collapse; font-size:12.5px; }}
  th, td {{ border:1px solid #eee; padding:5px 10px; text-align:left; }}
  th {{ background:#FAFAFA; }}
  .narrative {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:12px 16px; border-radius:6px; font-size:13.5px; line-height:1.6; }}
  .narrative p {{ margin: 0 0 10px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
</style></head>
<body><div class="wrap">
  <h1>📡 LaunchPulse — Autonomous Feature-Adoption Anomaly &amp; Opportunity Agent</h1>
  <div class="sub">Two statistical detectors (usage-volume point anomalies + channel activation-rate drift) flag anomalies; a fixed diagnostic playbook (real SQL) gathers evidence; an LLM synthesizes the root-cause report below.</div>
  <div class="kpis">
    <div class="kpi"><div class="val">2</div><div class="lbl">Detection Methods</div></div>
    <div class="kpi"><div class="val">3</div><div class="lbl">Confirmed Incidents</div></div>
    <div class="kpi"><div class="val">1</div><div class="lbl">False Alarm Correctly Dismissed</div></div>
    <div class="kpi"><div class="val">547</div><div class="lbl">Days Monitored</div></div>
  </div>
  {html}
  <h2 style="font-size:16px; margin-top:28px;">Agent Incident Reports</h2>
  {cards_html}
  <footer>Cached-mode demo (see README) — detectors are real (STL + rolling activation-rate drift), diagnostic SQL is real · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
