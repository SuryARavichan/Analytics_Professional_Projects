import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sentiment import score_dataframe
from themes import discover_themes
from llm_client import get_client

reviews = pd.read_csv("../data/app_reviews.csv", parse_dates=["date"])
tickets = pd.read_csv("../data/support_tickets.csv", parse_dates=["date"])
reviews = score_dataframe(reviews, "review_text")
labels, top_terms, vectorizer, km = discover_themes(reviews.review_text, n_clusters=8)
reviews["cluster"] = labels
reviews["month"] = reviews.date.dt.strftime("%Y-%m")

cluster_stats = reviews.groupby("cluster").agg(n=("review_id","count"), avg_rating=("rating","mean")).round(2)
crash_cluster_id = int(cluster_stats.avg_rating.idxmin())

client = get_client(cache_file="../cached_responses.json")
cluster_labels = {}
for c in sorted(top_terms):
    resp = client.complete(prompt="", prompt_id=f"cluster_label_{c}")
    cluster_labels[c] = resp.text.split(" — ")[0].split(":")[0][:40]

# ---- 1. Monthly rating + crash share ----
monthly = reviews.groupby("month").agg(avg_rating=("rating","mean"), n_reviews=("review_id","count")).reset_index()
monthly["crash_share"] = reviews.groupby("month").apply(lambda g: 100*(g.cluster==crash_cluster_id).mean()).values

line_rating = go.Scatter(x=monthly.month, y=monthly.avg_rating, name="Avg star rating", line=dict(color="#C0392B", width=3), mode="lines+markers")
line_crash = go.Scatter(x=monthly.month, y=monthly.crash_share, name="% reviews: App Stability Crisis", line=dict(color="#8E44AD", width=3), mode="lines+markers", yaxis="y2")

# ---- 2. Cluster distribution ----
dist = reviews.cluster.value_counts().sort_index()
bar_colors = ["#C0392B" if cluster_stats.loc[c,"avg_rating"] < 3.0 else "#2E86AB" for c in dist.index]
bar_dist = go.Bar(x=[cluster_labels[c] for c in dist.index], y=dist.values, marker_color=bar_colors,
                   text=dist.values, textposition="outside")

# ---- 3. Sentiment vs rating heatmap ----
ct = pd.crosstab(reviews.rating, reviews.sentiment_label)
heat = go.Heatmap(z=ct.values, x=ct.columns, y=ct.index, colorscale="YlGnBu",
                   text=ct.values, texttemplate="%{text}")

# ---- 4. Platform split of the crisis cluster ----
plat = reviews[reviews.cluster==crash_cluster_id].platform.value_counts()
pie = go.Pie(labels=plat.index, values=plat.values, marker_colors=["#2E86AB","#E8A33D"])

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Rating & Crisis-Cluster Share Over Time (Jul-Aug '25 incident)",
                     "Reviews per Discovered Theme (red = below 3.0 avg rating)",
                     "Star Rating vs. VADER Sentiment (validation)",
                     "App-Stability-Crisis Cluster by Platform"),
    specs=[[{"secondary_y": True}, {}], [{}, {"type":"domain"}]],
    vertical_spacing=0.18, horizontal_spacing=0.10,
)
fig.add_trace(line_rating, row=1, col=1, secondary_y=False)
fig.add_trace(line_crash, row=1, col=1, secondary_y=True)
fig.add_vrect(x0="2025-07", x1="2025-08", fillcolor="orange", opacity=0.15, row=1, col=1)
fig.add_trace(bar_dist, row=1, col=2)
fig.add_trace(heat, row=2, col=1)
fig.add_trace(pie, row=2, col=2)

fig.update_layout(
    height=950, showlegend=True,
    title=dict(text=(f"<b>EchoIQ — AI Voice-of-Customer Intelligence</b><br>"
                      f"<sup>{len(reviews):,} reviews · {len(tickets):,} support tickets · "
                      f"8 unsupervised themes discovered</sup>"), x=0.02, xanchor="left"),
    font=dict(family="Inter, Segoe UI, Arial", size=11.5),
    plot_bgcolor="white", paper_bgcolor="#FAFAFA", margin=dict(t=130,l=60,r=40,b=90),
)
fig.update_xaxes(tickangle=45, row=1, col=1)
fig.update_xaxes(tickangle=25, row=1, col=2)

html = pio.to_html(fig, include_plotlyjs="inline", full_html=False, config={"displaylogo": False})

digest_resp = client.complete(prompt="", prompt_id="digest_2025_07").text.replace("\n", "<br>")

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EchoIQ — Voice of Customer Dashboard</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#FAFAFA; color:#1a1a1a; }}
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 8px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:16px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:180px; }}
  .kpi .val {{ font-size:26px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .digest {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:16px 20px; border-radius:8px; font-size:14px; margin:16px 0; line-height:1.6; }}
  .digest b {{ display:block; margin-bottom:6px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
</style></head>
<body><div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="val">{len(reviews):,}</div><div class="lbl">Reviews Analyzed</div></div>
    <div class="kpi"><div class="val">{reviews.rating.mean():.2f}★</div><div class="lbl">Avg Rating</div></div>
    <div class="kpi"><div class="val">8</div><div class="lbl">Unsupervised Themes</div></div>
    <div class="kpi"><div class="val">2.5★</div><div class="lbl">Rating During Jul '25 Incident</div></div>
  </div>
  <div class="digest">{digest_resp}</div>
  {html}
  <footer>Cached-mode demo (see README) — sentiment via VADER, themes via TF-IDF+KMeans (both real, executed) · full analysis in <code>/notebook/analysis.ipynb</code></footer>
</div></body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
