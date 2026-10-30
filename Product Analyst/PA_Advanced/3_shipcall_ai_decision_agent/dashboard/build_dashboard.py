"""
Builds ShipCall AI's self-contained HTML dashboard: KPI cards, an
interactive Plotly overview of all 5 decisions, and the 5 full decision
memos rendered as cards with their supporting diagnostic evidence.
"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from agent import review_experiment
from llm_client import get_client

experiments = json.load(open("../data/experiments.json"))["experiments"]
client = get_client(cache_file="../cached_responses.json")
memos = {e["experiment_id"]: review_experiment(e, client) for e in experiments}

decision_colors = {"SHIP": "#2E86AB", "HOLD": "#E8A33D", "ITERATE": "#8E44AD", "INVALID": "#7F8C8D", "KILL": "#C0392B"}

summary_rows = []
for eid, memo in memos.items():
    decision = memo.narrative.split("\n")[0].replace("RECOMMENDATION: ", "").strip()
    summary_rows.append({
        "experiment_id": eid, "name": memo.name, "decision": decision,
        "lift_pp": memo.primary_result["lift_pp"], "p_value": memo.primary_result["p_value"],
        "srm": memo.srm.is_srm, "guardrail_violated": any(g.violated for g in memo.guardrail_results),
        "hte": memo.hte.any_significant_interaction,
    })
summary_df = pd.DataFrame(summary_rows)

# ---- overview chart ----
fig = go.Figure()
fig.add_trace(go.Bar(
    x=summary_df.lift_pp, y=summary_df.name, orientation="h",
    marker_color=[decision_colors[d] for d in summary_df.decision],
    text=summary_df.decision, textposition="outside",
    hovertext=[f"lift {l:+.2f}pp, p={p:.4f}" for l, p in zip(summary_df.lift_pp, summary_df.p_value)],
    hoverinfo="text",
))
fig.add_vline(x=0, line_color="black", line_width=1)
fig.update_layout(
    height=380, title="Primary Metric Lift vs. Final Decision — Notice How Often They Diverge",
    xaxis_title="Primary metric lift (pp)",
    font=dict(family="Inter, Segoe UI, Arial", size=12),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=60, l=40, r=100, b=50),
)
html_chart = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})


def evidence_line(memo):
    bits = [f"SRM: {'DETECTED' if memo.srm.is_srm else 'clean'} (p={memo.srm.p_value:.2e})"]
    viol = [g for g in memo.guardrail_results if g.violated]
    bits.append(f"Guardrails: {'VIOLATED (' + ', '.join(g.name for g in viol) + ')' if viol else 'clean'}")
    bits.append(f"HTE: {'significant segment effect' if memo.hte.any_significant_interaction else 'none detected'}")
    return " · ".join(bits)


cards_html = ""
for eid, memo in memos.items():
    decision = memo.narrative.split("\n")[0].replace("RECOMMENDATION: ", "").strip()
    narrative_html = memo.narrative.split("\n", 1)[1].strip().replace("\n\n", "</p><p>").replace("\n", "<br>")
    seg_rows = "".join(
        f"<tr><td>{r.segment}</td><td>{r.control_rate:.3f}</td><td>{r.treatment_rate:.3f}</td><td>{r.lift_pp:+.2f}pp</td></tr>"
        for _, r in memo.hte.segment_effects.iterrows()
    )
    cards_html += f"""
    <div class="card">
      <div class="card-header"><h3>{memo.name}</h3><span class="badge" style="background:{decision_colors[decision]}">{decision}</span></div>
      <div class="card-sub">{evidence_line(memo)}</div>
      <div class="evidence">
        <table><tr><th>Segment</th><th>Control</th><th>Treatment</th><th>Lift</th></tr>{seg_rows}</table>
      </div>
      <div class="narrative"><p>{narrative_html}</p></div>
    </div>"""

decisions_count = summary_df.decision.value_counts()

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ShipCall AI — Experiment Decision Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#F4F6F8; color:#1a1a1a; }}
  .wrap {{ max-width: 1250px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .sub {{ color:#666; font-size:14px; margin-bottom: 18px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:14px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:160px; }}
  .kpi .val {{ font-size:22px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:11px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .panel {{ background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:8px 16px; margin-bottom:20px; }}
  .card {{ background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:20px 24px; margin-bottom:20px; }}
  .card-header {{ display:flex; align-items:center; gap:12px; }}
  .card-header h3 {{ margin:0; font-size:16px; }}
  .card-sub {{ font-size:12px; color:#888; margin: 4px 0 12px; }}
  .badge {{ padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; color:white; }}
  .evidence {{ margin-bottom: 14px; }}
  table {{ border-collapse:collapse; font-size:12.5px; }}
  th, td {{ border:1px solid #eee; padding:5px 10px; text-align:left; }}
  th {{ background:#FAFAFA; }}
  .narrative {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:12px 16px; border-radius:6px; font-size:13.5px; line-height:1.6; }}
  .narrative p {{ margin: 0 0 10px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
</style></head>
<body><div class="wrap">
  <h1>🚦 ShipCall AI — Autonomous Experiment Readout &amp; Decision Agent</h1>
  <div class="sub">A fixed diagnostic playbook (SRM check → guardrail check → heterogeneous-treatment-effect test) runs before an LLM applies an explicit Ship/Hold/Iterate/Kill decision rubric to each completed experiment.</div>
  <div class="kpis">
    <div class="kpi"><div class="val">5</div><div class="lbl">Experiments Reviewed</div></div>
    <div class="kpi"><div class="val">{decisions_count.get('SHIP',0)}</div><div class="lbl">SHIP</div></div>
    <div class="kpi"><div class="val">{decisions_count.get('HOLD',0)}</div><div class="lbl">HOLD (guardrail)</div></div>
    <div class="kpi"><div class="val">{decisions_count.get('ITERATE',0)}</div><div class="lbl">ITERATE (segment HTE)</div></div>
    <div class="kpi"><div class="val">{decisions_count.get('INVALID',0)}</div><div class="lbl">INVALID (SRM)</div></div>
  </div>
  <div class="panel">{html_chart}</div>
  <h2 style="font-size:16px; margin-top:28px;">Full Decision Memos</h2>
  {cards_html}
  <footer>Cached-mode demo (see README) — SRM, guardrail, and HTE checks are real, executed statistics · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
