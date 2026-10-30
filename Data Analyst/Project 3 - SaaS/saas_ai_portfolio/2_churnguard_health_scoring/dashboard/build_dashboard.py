"""
Builds ChurnGuard AI's self-contained HTML dashboard: KPI cards, an
interactive 2x2 health-quadrant scatter (hover = account detail), a
feature-importance comparison chart, and the AI-generated renewal briefs
for the 10 highest-priority upcoming renewals, rendered as cards.
"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from health_scoring import prepare_features, train_model, assign_quadrant
from llm_client import get_client

conn = sqlite3.connect("../data/churnguard.db")
accounts = pd.read_sql_query("SELECT * FROM accounts", conn)
snapshot = pd.read_sql_query(open("../sql/02_feature_engineering.sql").read(), conn)

df_model, feature_cols = prepare_features(snapshot)
churn_clf, churn_auc = train_model(df_model, feature_cols, "churned")
expansion_clf, expansion_auc = train_model(df_model, feature_cols, "expanded")
df_model["churn_risk_score"] = churn_clf.predict_proba(df_model[feature_cols])[:, 1]
df_model["expansion_score"] = expansion_clf.predict_proba(df_model[feature_cols])[:, 1]
df_model["account_id"] = snapshot["account_id"].values
df_model["quadrant"] = [assign_quadrant(r, e) for r, e in zip(df_model.churn_risk_score, df_model.expansion_score)]

merged = snapshot.merge(df_model[["account_id", "churn_risk_score", "expansion_score", "quadrant"]], on="account_id")
active = merged[merged.churned == 0].copy()

colors = {"At Risk — Save": "#C0392B", "Growth Opportunity": "#2E86AB",
          "Engaged but Unstable": "#E8A33D", "Stable — Maintain": "#95A5A6"}

# ---- 1. Quadrant scatter ----
scatter_traces = []
for q, sub in active.groupby("quadrant"):
    hover = [f"{r.account_id} · {r.industry}<br>MRR ${r.mrr:,.0f} · {r.plan_tier}/{r.contract_type}<br>"
             f"Utilization {r.utilization_pct:.1f}% · Churn risk {r.churn_risk_score:.0%} · Expansion {r.expansion_score:.0%}"
             for _, r in sub.iterrows()]
    scatter_traces.append(go.Scatter(
        x=sub.churn_risk_score, y=sub.expansion_score, mode="markers", name=f"{q} (n={len(sub)})",
        marker=dict(size=(sub.mrr / 350).clip(lower=4), color=colors[q], opacity=0.55, line=dict(width=0)),
        text=hover, hoverinfo="text",
    ))

# ---- 2. Feature importance ----
churn_imp = pd.Series(churn_clf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(8)
exp_imp = pd.Series(expansion_clf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(8)

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.6, 0.4],
    subplot_titles=("Account Health Quadrant (bubble size = MRR)", "Top Feature Importances — Churn Model"),
)
for tr in scatter_traces:
    fig.add_trace(tr, row=1, col=1)
fig.add_shape(type="line", x0=0.35, x1=0.35, y0=0, y1=1, line=dict(color="grey", dash="dash", width=1), row=1, col=1)
fig.add_shape(type="line", x0=0, x1=1, y0=0.35, y1=0.35, line=dict(color="grey", dash="dash", width=1), row=1, col=1)
fig.add_trace(go.Bar(x=churn_imp.values, y=churn_imp.index, orientation="h", marker_color="#C0392B", showlegend=False), row=1, col=2)

fig.update_xaxes(title_text="Churn risk score", row=1, col=1)
fig.update_yaxes(title_text="Expansion score", row=1, col=1)
fig.update_layout(
    height=560, legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
    font=dict(family="Inter, Segoe UI, Arial", size=11.5),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=60, l=60, r=30, b=90),
)
html_chart = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})

# ---- 3. Quadrant summary table ----
qsum = active.groupby("quadrant").agg(accounts=("account_id", "count"), total_mrr=("mrr", "sum")).round(0)
qsum = qsum.reindex(["At Risk — Save", "Engaged but Unstable", "Growth Opportunity", "Stable — Maintain"])
qrows = "".join(
    f'<tr><td><span class="dot" style="background:{colors[q]}"></span>{q}</td>'
    f'<td>{int(r.accounts):,}</td><td>${r.total_mrr:,.0f}</td></tr>'
    for q, r in qsum.iterrows()
)

# ---- 4. Renewal briefs ----
brief_client = get_client(cache_file="../cached_responses.json")
upcoming = active[(active.days_to_renewal >= 0) & (active.days_to_renewal <= 90)].copy()

at_risk_priority = upcoming[upcoming.quadrant == "At Risk — Save"].copy()
at_risk_priority["priority"] = at_risk_priority.mrr * at_risk_priority.churn_risk_score
top_at_risk = at_risk_priority.sort_values("priority", ascending=False).head(5)

growth_priority = upcoming[upcoming.quadrant == "Growth Opportunity"].copy()
growth_priority["priority"] = growth_priority.mrr * growth_priority.expansion_score
top_growth = growth_priority.sort_values("priority", ascending=False).head(5)


def brief_cards(df, badge_color):
    cards = ""
    for _, acc in df.iterrows():
        resp = brief_client.complete(prompt=f"Generate renewal brief for {acc.account_id}", prompt_id=f"brief_{acc.account_id}")
        text = resp.text.replace("\n\n", "</p><p>").replace("\n", "<br>")
        cards += (
            f'<div class="brief-card"><div class="brief-head">'
            f'<span class="badge" style="background:{badge_color}">{acc.account_id}</span>'
            f'<span class="brief-mrr">${acc.mrr:,.0f} MRR · renews in {int(acc.days_to_renewal)}d</span></div>'
            f'<p>{text}</p></div>'
        )
    return cards


at_risk_html = brief_cards(top_at_risk, "#C0392B")
growth_html = brief_cards(top_growth, "#2E86AB")

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ChurnGuard AI — Health Scoring Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#FAFAFA; color:#1a1a1a; }}
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .sub {{ color:#666; font-size:14px; margin-bottom: 18px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:16px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:180px; }}
  .kpi .val {{ font-size:26px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .panel {{ background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:8px 16px; margin-bottom:20px; }}
  table.qtable {{ border-collapse:collapse; font-size:13px; width:100%; max-width:520px; margin: 10px 0 18px; }}
  table.qtable td {{ padding:8px 12px; border-bottom:1px solid #eee; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }}
  h2 {{ font-size:16px; margin: 28px 0 4px; }}
  .briefsub {{ color:#666; font-size:13px; margin-bottom:12px; }}
  .brief-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:24px; }}
  .brief-card {{ background:white; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:16px 18px; font-size:13px; line-height:1.55; }}
  .brief-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .badge {{ color:white; font-size:11px; font-weight:700; padding:3px 9px; border-radius:10px; }}
  .brief-mrr {{ color:#888; font-size:11.5px; }}
  .brief-card p {{ margin:0; white-space: pre-wrap; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
  @media (max-width: 900px) {{ .brief-grid {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🛡️ ChurnGuard AI — Two-Sided Health Scoring &amp; Renewal Briefs</h1>
  <div class="sub">2,800 B2B SaaS accounts · churn model AUC {churn_auc:.2f} · expansion model AUC {expansion_auc:.2f} · leakage-free features (window-function snapshots)</div>
  <div class="kpis">
    <div class="kpi"><div class="val">{accounts.churned.mean():.1%}</div><div class="lbl">Overall Churn Rate</div></div>
    <div class="kpi"><div class="val">{accounts.expanded.mean():.1%}</div><div class="lbl">Expansion-Candidate Rate</div></div>
    <div class="kpi"><div class="val">${accounts.mrr.sum():,.0f}</div><div class="lbl">Total MRR</div></div>
    <div class="kpi"><div class="val">{len(upcoming):,}</div><div class="lbl">Active Renewals in 90 Days</div></div>
  </div>
  <div class="panel">
    {html_chart}
    <table class="qtable">
      <tr><td><b>Quadrant</b></td><td><b>Accounts</b></td><td><b>Total MRR</b></td></tr>
      {qrows}
    </table>
  </div>

  <h2>🔴 AI-Generated Renewal Briefs — Top 5 "At Risk — Save" (by MRR × churn risk)</h2>
  <div class="briefsub">Routed to Customer Success for save-motion outreach this week.</div>
  <div class="brief-grid">{at_risk_html}</div>

  <h2>🟦 AI-Generated Renewal Briefs — Top 5 "Growth Opportunity" (by MRR × expansion score)</h2>
  <div class="briefsub">Routed to Sales for seat-expansion conversations at renewal.</div>
  <div class="brief-grid">{growth_html}</div>

  <footer>Cached-mode demo (see README) — both RandomForest models and all SQL are real and executed · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
