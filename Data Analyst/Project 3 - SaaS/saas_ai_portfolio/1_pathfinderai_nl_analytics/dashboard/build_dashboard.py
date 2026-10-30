"""
Builds an interactive, click-through "chat" demo of PathfinderAI as a
self-contained HTML page. No server needed -- the 12 demo Q&A pairs
(question -> generated SQL -> query result -> AI insight) are precomputed
and embedded as JSON; a little vanilla JS renders the chat interaction
client-side when you click a suggested question.
"""
import json

with open("demo_qa.json") as f:
    demo = json.load(f)
with open("../eval/eval_summary.json") as f:
    eval_summary = json.load(f)

demo_json = json.dumps(demo)

buttons_html = "".join(
    f'<button class="qbtn" data-idx="{i}">{d["question"]}</button>' for i, d in enumerate(demo)
)

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PathfinderAI — SaaS Analytics Copilot Demo</title>
<style>
  body {{ font-family: Inter, 'Segoe UI', Arial, sans-serif; margin:0; background:#F4F6F8; color:#1a1a1a; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .sub {{ color:#666; font-size:14px; margin-bottom: 18px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .kpi {{ background:white; border-radius:10px; padding:14px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:160px; }}
  .kpi .val {{ font-size:22px; font-weight:700; color:#2E86AB; }}
  .kpi .lbl {{ font-size:11px; color:#666; text-transform:uppercase; letter-spacing:0.03em; }}
  .layout {{ display:flex; gap:20px; align-items:flex-start; }}
  .sidebar {{ width: 300px; flex-shrink:0; }}
  .sidebar h3 {{ font-size:13px; text-transform:uppercase; color:#888; margin: 4px 0 8px; }}
  .qbtn {{ display:block; width:100%; text-align:left; background:white; border:1px solid #e2e2e2; border-radius:8px;
           padding:10px 12px; margin-bottom:8px; font-size:13px; cursor:pointer; transition: all .15s; }}
  .qbtn:hover {{ border-color:#2E86AB; background:#EAF4FA; }}
  .qbtn.active {{ background:#2E86AB; color:white; border-color:#2E86AB; }}
  .chat {{ flex:1; background:white; border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,0.08); padding:24px; min-height: 480px; }}
  .bubble-user {{ background:#2E86AB; color:white; padding:10px 16px; border-radius:14px 14px 4px 14px; display:inline-block; max-width:85%; margin-bottom:16px; font-size:14px; }}
  .stage {{ margin-bottom: 16px; }}
  .stage-label {{ font-size:11px; text-transform:uppercase; letter-spacing:0.04em; color:#999; margin-bottom:4px; font-weight:600; }}
  pre {{ background:#1e1e2e; color:#cdd6f4; padding:12px 14px; border-radius:8px; font-size:12.5px; overflow-x:auto; margin:0; }}
  table {{ border-collapse:collapse; font-size:13px; margin-top:4px; }}
  th, td {{ border:1px solid #e2e2e2; padding:6px 12px; text-align:left; }}
  th {{ background:#F4F6F8; }}
  .insight-box {{ background:#FFF7E6; border-left:4px solid #E8A33D; padding:12px 16px; border-radius:6px; font-size:14px; }}
  .placeholder {{ color:#999; text-align:center; padding: 100px 0; font-size:14px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; background:#EAF4FA; color:#2E86AB; margin-left:6px; }}
  footer {{ font-size:12px; color:#888; text-align:center; padding: 24px 0; }}
</style></head>
<body><div class="wrap">
  <h1>🧭 PathfinderAI — Natural-Language Product Analytics Copilot</h1>
  <div class="sub">Click a question to see the full pipeline: NL question → generated SQL → executed result → AI-written insight.</div>
  <div class="kpis">
    <div class="kpi"><div class="val">{eval_summary['overall_accuracy']:.0%}</div><div class="lbl">Eval Accuracy (20 Qs)</div></div>
    <div class="kpi"><div class="val">4</div><div class="lbl">Joined Tables</div></div>
    <div class="kpi"><div class="val">100%</div><div class="lbl">Unsafe Queries Blocked</div></div>
    <div class="kpi"><div class="val">3.2K</div><div class="lbl">Accounts Queryable</div></div>
  </div>
  <div class="layout">
    <div class="sidebar">
      <h3>Try a question</h3>
      {buttons_html}
    </div>
    <div class="chat" id="chat">
      <div class="placeholder">👈 Click a question to run it through PathfinderAI</div>
    </div>
  </div>
  <footer>Cached-mode demo (see README) — full pipeline code in <code>/pipeline</code>, eval harness in <code>/eval</code>, executed notebook in <code>/notebook/analysis.ipynb</code></footer>
</div>
<script>
const DEMO = {demo_json};

function fmtRow(r) {{ return r.map(v => typeof v === 'number' ? v.toLocaleString(undefined, {{maximumFractionDigits:2}}) : v); }}

function render(idx) {{
  const d = DEMO[idx];
  document.querySelectorAll('.qbtn').forEach((b,i) => b.classList.toggle('active', i===idx));
  let rowsHtml = '<table><tr>' + d.columns.map(c => `<th>${{c}}</th>`).join('') + '</tr>';
  d.rows.forEach(r => {{ rowsHtml += '<tr>' + fmtRow(r).map(v => `<td>${{v}}</td>`).join('') + '</tr>'; }});
  rowsHtml += '</table>';

  document.getElementById('chat').innerHTML = `
    <div class="bubble-user">${{d.question}}</div>
    <div class="stage"><div class="stage-label">Generated SQL <span class="badge">schema-aware · safety-checked</span></div>
      <pre>${{d.sql.replace(/</g,'&lt;')}}</pre></div>
    <div class="stage"><div class="stage-label">Query Result (${{d.rows.length}} row${{d.rows.length===1?'':'s'}})</div>${{rowsHtml}}</div>
    <div class="stage"><div class="stage-label">AI-Generated Insight</div><div class="insight-box">${{d.insight}}</div></div>
  `;
}}

document.querySelectorAll('.qbtn').forEach((b,i) => b.addEventListener('click', () => render(i)));
render(0);
</script>
</body></html>"""

with open("index.html", "w") as f:
    f.write(page)
print("Wrote dashboard/index.html —", len(page), "bytes")
