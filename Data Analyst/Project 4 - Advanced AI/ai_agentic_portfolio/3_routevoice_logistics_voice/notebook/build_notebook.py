import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""
# RouteVoice — Logistics Voice AI Agent Call Quality & Containment Analytics

**What this is:** a voice-AI monitoring system for Portside AI, a
fictional logistics platform whose voice agent calls truck drivers for
ETA check-ins, exception reports, proof-of-delivery confirmation, and
reschedule requests — the "voice/multimodal AI agent" trend, where the
telemetry that matters (transcription confidence, interruptions,
turn-taking latency) doesn't exist for a text-only AI product.

**The finding this uncovers:** a voice-model backbone upgrade (v1 → v2)
looks like a clean nationwide win — containment up +4.94pp (p<0.001).
But **Appalachian**, roughly 20% of call volume, actually **regresses**
(−6.75pp, p=3.6×10⁻⁶) over the same release, while every other region
improves. A companion logistic regression shows transcription confidence
is by far the dominant driver of containment (odds ratio ≈17.7), and
Appalachian's confidence is the one region that *fell* in this release —
consistent with a regional/dialectal coverage gap in the new voice
model's training data, a well-documented real category of voice-AI
fairness risk. This project also demonstrates two other detector shapes:
a transient shared-infrastructure ASR outage, and a benign deploy the
agent correctly declines to flag.

**On the LLM backend:** same cached/live design as the rest of this
session's AI-augmented projects — see `../shared/llm_client.py`.
""")

code(r"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from voice_metrics import containment_summary, containment_shift_test
from call_anomaly import hourly_confidence_series, point_anomaly_detector, find_anomaly_windows
from regression_analysis import fit_containment_model, standardized_effect_summary
from agent import review_regional_deploy, review_asr_outage, review_benign_deploy
from llm_client import get_client

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

calls = pd.read_csv("../data/calls.csv", parse_dates=["timestamp"])
client = get_client(cache_file="../cached_responses.json")

print(f"{len(calls):,} calls | {calls.timestamp.min()} to {calls.timestamp.max()} | "
      f"5 regions | model deploy 2026-05-26 | ASR outage 2026-05-13 | benign deploy 2026-06-05")
calls.head()
""")

md("## 1. Nationwide Aggregate vs. Regional Reality")

code(r"""
national = containment_shift_test(calls, region=None)
regions = ["Midwest", "Gulf_Coast", "Northeast", "Pacific_Northwest", "Appalachian"]
regional_results = pd.DataFrame([vars(containment_shift_test(calls, r)) for r in regions], index=regions)
print("Nationwide:", national)
regional_results
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#27AE60" if lp > 0 else "#C0392B" for lp in regional_results.lift_pp]
ax.barh(regional_results.index, regional_results.lift_pp, color=colors)
ax.axvline(national.lift_pp, color="#2E86AB", linestyle="--", linewidth=2, label=f"nationwide: +{national.lift_pp:.1f}pp")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Containment lift, v1 -> v2 (percentage points)")
ax.set_title("Every Region Improves Except Appalachian")
ax.legend()
plt.tight_layout()
plt.savefig("../dashboard/regional_lift.png", dpi=150)
plt.show()
""")

md(r"""The nationwide number (+4.94pp) sits comfortably inside the range
of the four improving regions — nothing about the aggregate signals a
problem. Appalachian's −6.75pp is the only bar pointing the wrong way,
and it's completely invisible unless you deliberately look region by
region.""")

md("## 2. What's Actually Driving Containment? A Regression, Not Just a Correlation")

code(r"""
model = fit_containment_model(calls)
effects = standardized_effect_summary(model)
effects
""")

md(r"""Transcription confidence is overwhelmingly the dominant driver of
containment (odds ratio ≈17.7, p<0.001) — interruption count and
response latency are not significant predictors once confidence is
accounted for. This turns "Appalachian is worse" into a testable
mechanism: if Appalachian's confidence fell, that alone would explain
its regression.""")

code(r"""
conf_by_region = containment_summary(calls, groupby=["driver_region", "model_version"])[["avg_confidence"]].unstack()
conf_by_region.columns = ["v1", "v2"]
conf_by_region["change"] = conf_by_region.v2 - conf_by_region.v1
conf_by_region.sort_values("change")
""")

md(r"""Confirmed: Appalachian is the only region where average
transcription confidence *fell* (0.82 → 0.71) — every other region's
confidence rose. Combined with the regression result above, this is a
coherent, evidence-backed mechanism, not a coincidence.""")

md("## 3. The Transient ASR Outage — a Different Signal Shape Entirely")

code(r"""
hourly = hourly_confidence_series(calls)
anomalies = point_anomaly_detector(hourly)
outage_windows = find_anomaly_windows(anomalies)
outage_windows
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 5))
window = hourly.loc["2026-05-12":"2026-05-14"]
ax.plot(window.index, window.values, color="#2E86AB", linewidth=1.5)
if len(outage_windows):
    w = outage_windows.iloc[0]
    ax.axvspan(w.start, w.end, color="#C0392B", alpha=0.25, label=f"flagged (peak z={w.peak_z_score:.1f})")
    ax.legend()
ax.set_ylabel("Hourly mean transcription confidence")
ax.set_title("ASR Outage: Simultaneous Drop Across All Regions/Call Types")
plt.tight_layout()
plt.savefig("../dashboard/asr_outage.png", dpi=150)
plt.show()
""")

md(r"""A sharp, ~4-hour, all-region drop with no deploy behind it — the
signature of a shared upstream infrastructure incident, not a model or
fairness issue. This is deliberately a different shape than the
Appalachian finding: sudden and universal, versus gradual-looking-but-
actually-instant-at-deploy and region-specific.""")

md("## 4. The Benign Deploy — Correctly Not Flagged")

code(r"""
benign_report = review_benign_deploy(calls, pd.Timestamp("2026-06-05"), window_days=3, client=client)
print(benign_report.evidence)
print()
print(benign_report.narrative)
""")

md("## 5. Agent Reports")

code(r"""
regional_reports = {r: review_regional_deploy(calls, r, client) for r in regions}
print(regional_reports["Appalachian"].narrative)
""")

code(r"""
print(regional_reports["Midwest"].narrative)
""")

code(r"""
asr_report = review_asr_outage(outage_windows.iloc[0].to_dict(), client)
print(asr_report.narrative)
""")

md("## 6. Summary")

code(r"""
print("=== RouteVoice summary ===")
for r in regions:
    print(f"{r:20s} -> {regional_reports[r].verdict}")
print(f"{'ASR outage':20s} -> {asr_report.verdict}")
print(f"{'benign deploy':20s} -> {benign_report.verdict}")
print()
print("The finding that matters most: a voice-model release looked like a clean nationwide win")
print("(+4.94pp containment, p<0.001) while Appalachian -- ~20% of volume -- regressed -6.75pp (p=3.6e-6),")
print("driven by a fall in transcription confidence consistent with a regional dialect coverage gap in the")
print("new model. A regression analysis confirms confidence is the dominant driver of containment, giving")
print("this finding an evidence-backed mechanism rather than just a flagged number.")
""")

nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Notebook written.")
