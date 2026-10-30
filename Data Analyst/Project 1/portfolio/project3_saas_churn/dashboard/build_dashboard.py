import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

conn = sqlite3.connect("../data/churn.db")
def run_sql(path):
    with open(path) as f:
        return pd.read_sql_query(f.read(), conn)

customers = pd.read_sql_query("SELECT * FROM customers", conn)
churn_rates = run_sql("../sql/01_churn_rates.sql")
trend = run_sql("../sql/03_monthly_churn_trend.sql")
scored = pd.read_csv("scored_accounts.csv")

total_accounts = len(customers)
overall_churn = customers.churned.mean() * 100
total_mrr = customers.initial_mrr.sum()
at_risk_mrr = scored[scored.risk_tier == "High"].initial_mrr.sum()

# ---- 1. Churn by plan/contract ----
pivot_logo = churn_rates.pivot(index="plan_tier", columns="contract_type", values="logo_churn_rate_pct").loc[["Starter","Growth","Enterprise"]]
bar_monthly = go.Bar(name="Monthly", x=pivot_logo.index, y=pivot_logo["Monthly"], marker_color="#C0392B")
bar_annual = go.Bar(name="Annual", x=pivot_logo.index, y=pivot_logo["Annual"], marker_color="#2E86AB")

# ---- 2. Monthly churn trend ----
line_trend = go.Scatter(x=trend.month, y=trend.monthly_logo_churn_rate_pct, mode="lines+markers",
                         line=dict(color="#C0392B", width=3))

# ---- 3. Risk tier distribution ----
risk_summary = scored.groupby("risk_tier", observed=True).agg(accounts=("customer_id","count"), mrr=("initial_mrr","sum")).reindex(["Low","Medium","High"])
bar_risk = go.Bar(x=risk_summary.index, y=risk_summary.accounts,
                   marker_color=["#2E86AB","#E8A33D","#C0392B"],
                   text=risk_summary.accounts, textposition="outside")

# ---- 4. Risk score distribution histogram ----
hist_risk = go.Histogram(x=scored.churn_risk_score, nbinsx=30, marker_color="#3B8686")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Logo Churn Rate by Plan & Contract Type",
        "Monthly Logo Churn Rate Trend",
        "Active Accounts by Churn-Risk Tier",
        "Churn-Risk Score Distribution (Active Accounts)"
    ),
    vertical_spacing=0.16, horizontal_spacing=0.10,
)
fig.add_trace(bar_monthly, row=1, col=1); fig.add_trace(bar_annual, row=1, col=1)
fig.add_trace(line_trend, row=1, col=2)
fig.add_trace(bar_risk, row=2, col=1)
fig.add_trace(hist_risk, row=2, col=2)

fig.update_layout(
    height=900, barmode="group", showlegend=True,
    title=dict(text=(f"<b>SaaS Churn Prediction & Diagnosis</b><br>"
                      f"<sup>{total_accounts:,} accounts · {overall_churn:.1f}% churned to date · "
                      f"${at_risk_mrr:,.0f} MRR in High-risk active accounts</sup>"), x=0.02, xanchor="left"),
    font=dict(family="Inter, Segoe UI, Arial", size=12),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=120,l=60,r=40,b=60),
)
fig.update_xaxes(tickangle=45, row=1, col=2)

html = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})

top_risk = scored.sort_values("churn_risk_score", ascending=False).head(15)
rows_html = "".join(
    f"<tr><td>{r.customer_id}</td><td>{r.plan_tier}</td><td>{r.contract_type}</td>"
    f"<td>{r.churn_risk_score:.0%}</td><td><span class='badge {r.risk_tier.lower()}'>{r.risk_tier}</span></td>"
    f"<td>${r.initial_mrr:,.0f}</td></tr>"
    for r in top_risk.itertuples()
)

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SaaS Churn Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#FAFAFA; color:#1a1a1a; }}
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 8px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:16px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:180px; }}
  .kpi .val {{ font-size:26px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .note {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:12px 16px; border-radius:6px; font-size:14px; margin:16px 0; }}
  table {{ width:100%; border-collapse:collapse; background:white; border-radius:10px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-top:8px;}}
  th, td {{ padding:10px 14px; text-align:left; border-bottom:1px solid #eee; font-size:13px; }}
  th {{ background:#2E4057; color:white; }}
  .badge {{ padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; color:white; }}
  .badge.high {{ background:#C0392B; }} .badge.medium {{ background:#E8A33D; }} .badge.low {{ background:#2E86AB; }}
  h3 {{ margin-top: 28px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
</style></head>
<body><div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="val">{total_accounts:,}</div><div class="lbl">Total Accounts</div></div>
    <div class="kpi"><div class="val">{overall_churn:.1f}%</div><div class="lbl">Cumulative Churn Rate</div></div>
    <div class="kpi"><div class="val">${total_mrr:,.0f}</div><div class="lbl">Total Initial MRR</div></div>
    <div class="kpi"><div class="val">${at_risk_mrr:,.0f}</div><div class="lbl">MRR in High-Risk Accounts</div></div>
  </div>
  <div class="note"><b>Key finding:</b> Annual contracts churn 2-3x less than Monthly within every plan tier.
  A Random Forest model using rolling engagement + support-ticket features reaches ~0.86 AUC predicting churn
  <i>before</i> it happens, using only pre-churn data.</div>
  {html}
  <h3>Top 15 Highest-Risk Active Accounts (for Customer Success outreach)</h3>
  <table><tr><th>Account</th><th>Plan</th><th>Contract</th><th>Churn Risk</th><th>Tier</th><th>MRR</th></tr>{rows_html}</table>
  <footer>Built with Python (scikit-learn) + Plotly from SQL queries in <code>/sql</code> · full analysis in <code>/notebook/analysis.ipynb</code> · full scored list in <code>scored_accounts.csv</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
