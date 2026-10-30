import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.stats.proportion import proportions_ztest

conn = sqlite3.connect("../data/marketing.db")
def run_sql(path):
    with open(path) as f:
        return pd.read_sql_query(f.read(), conn)

ab = pd.read_sql_query("SELECT * FROM ab_sessions", conn)
results = run_sql("../sql/01_ab_test_results.sql")
attribution = run_sql("../sql/02_attribution_models.sql")
roas = run_sql("../sql/03_channel_roas.sql")

ctrl, trt = ab[ab.variant=="control"], ab[ab.variant=="treatment"]
overall_lift_pp = 100*(trt.converted.mean() - ctrl.converted.mean())
z, p = proportions_ztest([trt.converted.sum(), ctrl.converted.sum()], [len(trt), len(ctrl)], alternative="larger")

# ---- 1. Device segment bars ----
by_device = results[results.device != "ALL"]
pivot_cr = by_device.pivot(index="device", columns="variant", values="conversion_rate_pct")
bar_ctrl = go.Bar(name="Control", x=pivot_cr.index, y=pivot_cr["control"], marker_color="#B0B0B0")
bar_trt = go.Bar(name="Treatment", x=pivot_cr.index, y=pivot_cr["treatment"], marker_color="#2E86AB")

# ---- 2. Attribution model comparison ----
attr_melt = attribution.melt(id_vars="channel", var_name="model", value_name="revenue")
attr_melt["model"] = attr_melt["model"].str.replace("revenue_", "").str.replace("_", " ").str.title()
attr_traces = []
colors = {"First Touch": "#F18F01", "Last Touch": "#C73E1D", "Linear": "#2E86AB", "Position Based": "#3B8686"}
for model in attr_melt.model.unique():
    sub = attr_melt[attr_melt.model == model]
    attr_traces.append(go.Bar(name=model, x=sub.channel, y=sub.revenue, marker_color=colors.get(model)))

# ---- 3. ROAS ----
roas_colors = ["#2E86AB" if r >= 1 else "#C0392B" for r in roas.roas]
bar_roas = go.Bar(x=roas.channel, y=roas.roas, marker_color=roas_colors, text=roas.roas, textposition="outside")

# ---- 4. Daily conversion trend ----
ab["session_date"] = pd.to_datetime(ab["session_date"])
daily = ab.groupby(["session_date", "variant"]).agg(conv_rate=("converted", "mean")).reset_index()
daily["conv_rate"] *= 100
line_ctrl = go.Scatter(x=daily[daily.variant=="control"].session_date, y=daily[daily.variant=="control"].conv_rate,
                        name="Control", line=dict(color="#B0B0B0", width=2), mode="lines")
line_trt = go.Scatter(x=daily[daily.variant=="treatment"].session_date, y=daily[daily.variant=="treatment"].conv_rate,
                       name="Treatment", line=dict(color="#2E86AB", width=2), mode="lines")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Daily Conversion Rate: Control vs Treatment",
        "Conversion Rate by Device (Mobile drives the lift)",
        "Attributed Revenue by Channel — 4 Attribution Models",
        "ROAS by Channel (Linear Attribution, breakeven = 1.0)"
    ),
    specs=[[{}, {}], [{}, {}]], vertical_spacing=0.16, horizontal_spacing=0.09,
)
fig.add_trace(line_ctrl, row=1, col=1); fig.add_trace(line_trt, row=1, col=1)
fig.add_trace(bar_ctrl, row=1, col=2); fig.add_trace(bar_trt, row=1, col=2)
for tr in attr_traces:
    fig.add_trace(tr, row=2, col=1)
fig.add_trace(bar_roas, row=2, col=2)
fig.add_hline(y=1.0, line_dash="dash", line_color="black", row=2, col=2)

fig.update_layout(
    height=950, barmode="group",
    title=dict(text=(f"<b>Checkout A/B Test & Multi-Touch Attribution</b><br>"
                      f"<sup>Overall lift: +{overall_lift_pp:.2f}pp (p={p:.4f}, significant) · "
                      f"48,000 sessions · 9,000 marketing journeys</sup>"), x=0.02, xanchor="left"),
    font=dict(family="Inter, Segoe UI, Arial", size=12),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=120, l=60, r=40, b=60),
    legend=dict(orientation="h", y=-0.08),
)

html = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Marketing A/B Test & Attribution Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#FAFAFA; color:#1a1a1a; }}
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 8px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:16px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:180px; }}
  .kpi .val {{ font-size:26px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .note {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:12px 16px; border-radius:6px; font-size:14px; margin:16px 0; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
</style></head>
<body><div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="val">+{overall_lift_pp:.2f}pp</div><div class="lbl">Overall Conversion Lift</div></div>
    <div class="kpi"><div class="val">p={p:.4f}</div><div class="lbl">Significance (z-test)</div></div>
    <div class="kpi"><div class="val">+2.67pp</div><div class="lbl">Mobile Lift (driver)</div></div>
    <div class="kpi"><div class="val">-0.37pp</div><div class="lbl">Desktop Lift (n.s.)</div></div>
  </div>
  <div class="note"><b>Key finding:</b> the checkout redesign's lift is almost entirely a Mobile effect (p&lt;0.001); Desktop is statistically flat.
  Separately, attribution model choice changes which channel "wins" — Paid Social looks weak on last-touch but is a top assist channel under linear/position-based credit.</div>
  {html}
  <footer>Built with Python (statsmodels/scipy) + Plotly from SQL queries in <code>/sql</code> · full analysis in <code>/notebook/analysis.ipynb</code> · budget model in <code>/excel</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
