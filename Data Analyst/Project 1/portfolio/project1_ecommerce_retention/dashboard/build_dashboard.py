"""
Builds a single self-contained interactive HTML dashboard for the
E-commerce Retention project using Plotly. No external dependencies at
view-time (plotly.js is inlined), so this file can be double-clicked,
hosted on GitHub Pages, or dropped into a portfolio site as-is.

This is the "instant, no-license-needed" deliverable. The same underlying
tables (data/ecommerce.db) can be pointed at from Tableau or Power BI
Desktop to rebuild an equivalent live dashboard — see the project README
for a field-by-field guide.
"""
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

conn = sqlite3.connect("../data/ecommerce.db")

def run_sql(path):
    with open(path) as f:
        return pd.read_sql_query(f.read(), conn)

cohort = run_sql("../sql/02_cohort_retention.sql")
rfm = run_sql("../sql/03_rfm_segmentation.sql")
diag = run_sql("../sql/04_channel_diagnosis.sql")
customers = pd.read_sql_query("SELECT * FROM customers", conn)
orders = pd.read_sql_query("SELECT * FROM orders", conn)
orders["order_date"] = pd.to_datetime(orders["order_date"])

# ---- KPI numbers -----------------------------------------------------
total_customers = len(customers)
total_revenue = orders.loc[orders.order_status != "Cancelled", "order_value"].sum()
repeat_rate = round(100 * (orders[orders.order_status != "Cancelled"].groupby("customer_id").size() >= 2).mean(), 1)
avg_order_value = round(orders.loc[orders.order_status != "Cancelled", "order_value"].mean(), 2)

# ---- 1. Cohort heatmap ------------------------------------------------
pivot = cohort.pivot(index="cohort_month", columns="months_since_signup", values="retention_rate_pct")
pivot = pivot.sort_index(ascending=False)
heatmap = go.Heatmap(
    z=pivot.values, x=[f"M{c}" for c in pivot.columns], y=pivot.index,
    colorscale="YlGnBu", colorbar=dict(title="% active"),
    text=pivot.values, texttemplate="%{text:.0f}", hoverongaps=False,
)

# ---- 2. Repeat rate by channel ----------------------------------------
cust_orders = orders[orders.order_status != "Cancelled"].groupby("customer_id").agg(n_orders=("order_id", "count")).reset_index()
cust_full = customers.merge(cust_orders, on="customer_id", how="left")
cust_full["n_orders"] = cust_full["n_orders"].fillna(0)
cust_full["is_repeat"] = cust_full["n_orders"] >= 2
repeat_by_channel = cust_full.groupby("acquisition_channel").agg(
    customers=("customer_id", "count"), repeat=("is_repeat", "sum")
).reset_index()
repeat_by_channel["repeat_rate_pct"] = round(100 * repeat_by_channel.repeat / repeat_by_channel.customers, 1)
repeat_by_channel = repeat_by_channel.sort_values("repeat_rate_pct", ascending=True)

bar_channel = go.Bar(
    x=repeat_by_channel.repeat_rate_pct, y=repeat_by_channel.acquisition_channel,
    orientation="h", marker_color=repeat_by_channel.repeat_rate_pct,
    marker_colorscale="Teal", text=repeat_by_channel.repeat_rate_pct.astype(str) + "%",
    textposition="outside",
)

# ---- 3. RFM segment sizes -----------------------------------------------
seg_summary = rfm.groupby("rfm_segment").agg(customers=("customer_id", "count"), avg_monetary=("monetary", "mean")).reset_index()
seg_summary = seg_summary.sort_values("customers", ascending=False)
bar_rfm = go.Bar(
    x=seg_summary.rfm_segment, y=seg_summary.customers, marker_color="#2E86AB",
    text=seg_summary.customers, textposition="outside", name="customers",
)

# ---- 4. Incident diagnosis timeline -------------------------------------
diag_monthly = diag.groupby("order_month").agg(
    total_orders=("total_orders", "sum"), cancelled=("cancelled_orders", "sum"),
    avg_review=("avg_review_score", "mean")
).reset_index()
diag_monthly["cancel_rate_pct"] = round(100 * diag_monthly.cancelled / diag_monthly.total_orders, 1)

line_cancel = go.Scatter(x=diag_monthly.order_month, y=diag_monthly.cancel_rate_pct,
                          mode="lines+markers", name="Cancel rate %", line=dict(color="crimson", width=3))
line_review = go.Scatter(x=diag_monthly.order_month, y=diag_monthly.avg_review, mode="lines+markers",
                          name="Avg review score", line=dict(color="steelblue", width=3), yaxis="y2")

# ---- assemble figure ------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Monthly Cohort Retention Heatmap (% active by month-since-signup)",
        "Repeat-Purchase Rate by Acquisition Channel",
        "Customers per RFM Segment",
        "Cancellation Rate & Review Score Over Time (Aug-Sep '25 incident)"
    ),
    specs=[[{"type": "heatmap"}, {"type": "bar"}], [{"type": "bar"}, {"secondary_y": True}]],
    vertical_spacing=0.14, horizontal_spacing=0.10,
)
fig.add_trace(heatmap, row=1, col=1)
fig.add_trace(bar_channel, row=1, col=2)
fig.add_trace(bar_rfm, row=2, col=1)
fig.add_trace(line_cancel, row=2, col=2, secondary_y=False)
fig.add_trace(line_review, row=2, col=2, secondary_y=True)

fig.add_vrect(x0="2025-08", x1="2025-09", fillcolor="orange", opacity=0.15,
              annotation_text="Fulfillment incident", annotation_position="top left", row=2, col=2)

fig.update_layout(
    height=950, showlegend=False,
    title=dict(
        text=(f"<b>E-commerce Customer Retention & Cohort Analysis</b><br>"
              f"<sup>{total_customers:,} customers · ${total_revenue:,.0f} revenue · "
              f"{repeat_rate}% repeat-purchase rate · ${avg_order_value} avg order value</sup>"),
        x=0.02, xanchor="left"
    ),
    font=dict(family="Inter, Segoe UI, Arial", size=12),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA",
    margin=dict(t=120, l=60, r=40, b=60),
)
fig.update_xaxes(tickangle=45, row=2, col=2)

html = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>E-commerce Retention Dashboard</title>
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
    <div class="kpi"><div class="val">{total_customers:,}</div><div class="lbl">Total Customers</div></div>
    <div class="kpi"><div class="val">${total_revenue:,.0f}</div><div class="lbl">Total Revenue</div></div>
    <div class="kpi"><div class="val">{repeat_rate}%</div><div class="lbl">Repeat Purchase Rate</div></div>
    <div class="kpi"><div class="val">${avg_order_value}</div><div class="lbl">Avg Order Value</div></div>
  </div>
  <div class="note"><b>Key finding:</b> Referral & Organic customers retain ~2x better than Paid Social (chi-square p&lt;0.001).
  A fulfillment issue in Aug-Sep 2025 doubled cancellation rates and dropped review scores ~0.8pts, visibly denting retention for cohorts acquired in that window.</div>
  {html}
  <footer>Built with Python + Plotly from SQL (SQLite) queries in <code>/sql</code> · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)

print("Wrote dashboard/index.html —", len(page), "bytes")
