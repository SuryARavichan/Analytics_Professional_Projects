"""
Builds ResolveIQ's self-contained HTML dashboard: KPI cards, a 4-panel
Plotly view (official vs. true containment by category, the statistical
test comparison, the account_recovery cost/reopen-rate trade-off), and
all four category release-review reports rendered as cards, with
account_recovery's "false improvement" verdict front and center.
"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from containment import containment_summary, containment_shift_test, classify_policy_change
from economics import reopen_cost_impact
from agent import review_category
from llm_client import get_client

tickets = pd.read_csv("../data/tickets.csv", parse_dates=["timestamp"])
tickets["reopened_within_48h"] = tickets["reopened_within_48h"].astype(bool)
client = get_client(cache_file="../cached_responses.json")

categories = ["billing", "technical", "shipping", "account_recovery"]
rows = []
for version in ["v1", "v2"]:
    sub = tickets[tickets.agent_version == version]
    s = containment_summary(sub, groupby="category")
    s["agent_version"] = version
    rows.append(s.reset_index())
summary = pd.concat(rows).sort_values(["category", "agent_version"])

results = []
for cat in categories:
    off = containment_shift_test(tickets, cat, metric="official")
    true = containment_shift_test(tickets, cat, metric="true")
    verdict = classify_policy_change(off, true)
    results.append({"category": cat, "official_lift_pp": off.lift_pp, "official_p_value": off.p_value,
                     "true_lift_pp": true.lift_pp, "true_p_value": true.p_value, "verdict": verdict})
results_df = pd.DataFrame(results)

ar_cost = reopen_cost_impact(tickets, "account_recovery")
reports = {cat: review_category(tickets, cat, client) for cat in categories}

verdict_colors = {"GENUINE_IMPROVEMENT": "#27AE60", "FALSE_IMPROVEMENT": "#C0392B", "NO_CHANGE": "#95A5A6"}

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Official vs. True Containment Lift by Category (v1→v2)",
    "Statistical Significance: Official vs. True (−log10 p)",
    "account_recovery: Cost per Ticket vs. Reopen Rate",
    "Containment Rate by Category & Version",
))

fig.add_trace(go.Bar(x=results_df.category, y=results_df.official_lift_pp, name="official lift (pp)", marker_color="#2E86AB"), row=1, col=1)
fig.add_trace(go.Bar(x=results_df.category, y=results_df.true_lift_pp, name="true lift (pp)", marker_color="#C0392B"), row=1, col=1)
fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)

neglog_off = -np.log10(results_df.official_p_value.clip(lower=1e-300))
neglog_true = -np.log10(results_df.true_p_value.clip(lower=1e-300))
fig.add_trace(go.Bar(x=results_df.category, y=neglog_off, name="official −log10(p)", marker_color="#2E86AB", showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=results_df.category, y=neglog_true, name="true −log10(p)", marker_color="#C0392B", showlegend=False), row=1, col=2)
fig.add_hline(y=-np.log10(0.01), line_dash="dash", line_color="#333", row=1, col=2)

versions = ["v1", "v2"]
avg_cost = [ar_cost[v]["avg_cost_all"] for v in versions]
reopen_pct = [ar_cost[v]["reopen_rate"] * 100 for v in versions]
fig.add_trace(go.Bar(x=versions, y=avg_cost, name="avg cost/ticket ($)", marker_color="#2E86AB", showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=versions, y=reopen_pct, name="reopen rate (%)", mode="lines+markers",
                          line=dict(color="#C0392B", width=3), marker=dict(size=12), yaxis="y5"), row=2, col=1)

for version, color in [("v1", "#95A5A6"), ("v2", "#2E86AB")]:
    sub = summary[summary.agent_version == version]
    fig.add_trace(go.Bar(x=sub.category, y=sub.true_containment, name=f"{version} true containment", marker_color=color), row=2, col=2)

fig.update_yaxes(title_text="Lift (pp)", row=1, col=1)
fig.update_yaxes(title_text="-log10(p-value)", row=1, col=2)
fig.update_yaxes(title_text="Avg cost ($)", row=2, col=1)
fig.update_yaxes(title_text="True containment rate", row=2, col=2)
fig.update_layout(
    height=780, barmode="group",
    font=dict(family="Inter, Segoe UI, Arial", size=11.5),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=60, l=60, r=30, b=50),
    legend=dict(orientation="h", y=1.08),
)
html_chart = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})


def card(cat, verdict, narrative):
    color = verdict_colors.get(verdict, "#95A5A6")
    label = verdict.replace("_", " ").title()
    lines = narrative.strip().split("\n\n")
    body = "".join(f"<p>{l.replace(chr(10), '<br>')}</p>" for l in lines)
    return f"""<div class="card">
      <span class="badge" style="background:{color}">{label}</span>
      <h3>{cat.replace('_', ' ').title()}</h3>
      {body}
    </div>"""


cards_html = "".join(card(cat, reports[cat].verdict, reports[cat].narrative) for cat in categories)

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ResolveIQ — CX Agent Containment Dashboard</title>
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
  <h1>🤖 ResolveIQ — Autonomous CX Agent Performance &amp; Containment Analytics</h1>
  <div class="sub">Official (self-reported) vs. true (reopen-adjusted) containment for an autonomous AI support agent — catching an AI system's own success metric being gamed by a policy change.</div>
  <div class="kpis">
    <div class="kpi"><div class="val">+20.2pp</div><div class="lbl">account_recovery Official Containment "Improvement"</div></div>
    <div class="kpi"><div class="val">−0.04pp</div><div class="lbl">account_recovery TRUE Containment Change (p=0.98)</div></div>
    <div class="kpi"><div class="val">3.4x</div><div class="lbl">Reopen Rate Increase (9.8% → 33.9%)</div></div>
    <div class="kpi"><div class="val">2 / 4</div><div class="lbl">Categories Confirmed as Genuine Improvements</div></div>
  </div>
  <div class="panel">{html_chart}</div>
  <h2 style="font-size:16px;">Release-Review Reports by Category</h2>
  <div class="cards">{cards_html}</div>
  <footer>Synthetic ticket-level data, seeded and reproducible (see README) · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
