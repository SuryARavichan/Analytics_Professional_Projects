# RouteVoice — Logistics Voice AI Agent Call Quality & Containment Analytics

**One-line pitch:** a voice-AI monitoring system for a logistics voice-agent platform that catches a regional dialect coverage gap hiding inside a clean-looking nationwide release — a model upgrade that improves containment nationwide (+4.94pp) while one region actually regresses (−6.75pp, p=3.6×10⁻⁶), driven by a measurable drop in transcription confidence for that region alone.

## Why this project, and why voice AI needs its own analytics, not just text-LLM analytics

A voice AI agent produces telemetry a text-based AI product simply doesn't have: transcription confidence, interruption counts, turn-taking latency. Those signals matter because voice AI has a well-documented, real risk category text models don't: regional and dialectal speech-recognition performance gaps. This project builds voice-specific quality metrics, uses a regression model to establish which of them actually drives outcomes (not just correlates with them), and finds a real regional regression that a nationwide-only view would never catch — plus two other genuinely different incident shapes (a shared infrastructure outage, and a deploy that should be correctly ignored) to show the analytics distinguishes between them rather than treating every metric shift as the same kind of problem.

## What's actually built here (not just described)

- **`data/generate_data.py`** — 20,000 synthetic voice-agent calls for Portside AI across 5 US regions and 4 call types, with a voice-model backbone upgrade (v1→v2) that improves every region except Appalachian (a regional dialect coverage gap), a transient shared-ASR-provider outage, and a benign greeting-script deploy that changes call duration without touching containment.
- **`pipeline/voice_metrics.py`** — containment/confidence summaries and a two-proportion z-test usable on both the nationwide aggregate and any single region, so both can be tested with the same function and compared directly.
- **`pipeline/regression_analysis.py`** — a logistic regression of call containment on transcription confidence, interruption count, and response latency, giving an odds-ratio view of which mechanic actually drives outcomes rather than just observing that a region is "worse."
- **`pipeline/call_anomaly.py`** — an STL point-anomaly detector on hourly transcription confidence, for catching a shared-infrastructure incident that hits every region simultaneously — a different signal shape from a deploy-tied regional regression.
- **`pipeline/agent.py`** — perceive → act → reason → report across all three incident types, with an explicit rubric: a region-level regression is never excused by a healthy nationwide number, and a non-model deploy is only flagged if the outcome metric (containment) itself moved.
- **`notebook/analysis.ipynb`** — the full, executed walkthrough: nationwide-vs-regional comparison, the regression model, the ASR outage detection, the benign-deploy dismissal, and all agent reports. Zero execution errors (23 cells).
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards, a 4-panel Plotly view, and verdict-coded agent report cards.

## Key results

| Region | Containment lift (v1→v2) | p-value | Verdict |
|---|---|---|---|
| Midwest | +9.68pp | <0.001 | Genuine improvement |
| Pacific_Northwest | +7.71pp | 1.5×10⁻⁷ | Genuine improvement |
| Gulf_Coast | +6.40pp | 1.2×10⁻⁶ | Genuine improvement |
| Northeast | +5.92pp | 1.0×10⁻⁵ | Genuine improvement |
| **Appalachian** | **−6.75pp** | **3.6×10⁻⁶** | **Fairness/coverage risk** |
| **Nationwide aggregate** | **+4.94pp** | <0.001 | *Looks like a clean win* |

Logistic regression: transcription confidence is the dominant driver of containment (odds ratio ≈17.7, p<0.001); interruption count and response latency are not significant once confidence is controlled for. Appalachian's average transcription confidence fell from 0.82 to 0.71 across the same release — the only region where confidence dropped — giving the regional finding an evidence-backed mechanism, not just a flagged number.

The ASR outage (2026-05-13, 10:00-13:00, peak z-score ≈14.1) and the benign greeting-script deploy (call duration −17%, containment change not significant, p=0.95) round out the three incident types the agent correctly distinguishes.

## On the honest limitations (stated in the notebook, not hidden)

- **Region is a coarse proxy for dialect**, not a direct linguistic measurement — a real production system would want a more precise dialect/accent classification signal rather than inferring it from geography.
- **The regression model doesn't include a training-data-coverage measurement for the voice model itself** — the "training data underrepresentation" mechanism is the most plausible explanation given the evidence (confidence specifically dropping in the one regressing region), not a confirmed cause; confirming it would need access to the vendor's training data composition.
- **The ASR outage's root cause is inferred from its shape** (broad, simultaneous, no deploy tied to it), not confirmed via direct provider status logs, which this playbook doesn't have access to.

## Resume bullets

- Built a voice-AI monitoring system that caught a regional dialect coverage gap (Appalachian containment −6.75pp, p=3.6×10⁻⁶) completely hidden inside a nationwide release that looked like a clean win (+4.94pp, p<0.001) across the same deploy.
- Used logistic regression to establish transcription confidence as the dominant, statistically significant driver of call containment (odds ratio ≈17.7) among three candidate voice-AI telemetry signals, turning a flagged regional anomaly into an evidence-backed mechanism.
- Designed a 3-detector voice-AI monitoring pipeline (deploy-aware regional regression test, STL infrastructure-anomaly detector, benign-deploy dismissal test) and an agent rubric that never excuses a region-level regression with a healthy aggregate number.

## Interview talking points

- **"Why does voice AI need different analytics than text AI?"** — because it produces fundamentally different telemetry (transcription confidence, interruptions, turn-taking latency) tied to a real, distinct risk category: regional/dialectal speech-recognition performance gaps, which don't have a text-AI analog in the same form.
- **"How did you know confidence was the actual driver, not just a correlate?"** — a logistic regression controlling for interruption count and response latency simultaneously; confidence remained highly significant (p<0.001) while the other two candidates did not, which is what makes this a mechanism rather than a coincidence.
- **"How do you tell a fairness regression apart from an infra outage?"** — by shape and scope: the Appalachian regression is a step-change tied to a specific deploy and isolated to one region; the ASR outage is a sharp, simultaneous, all-region drop with no deploy behind it. Using the wrong lens on either would misdiagnose the cause and the fix.
- **"What would you change with more time/data?"** — replace the geography-based region proxy with an actual accent/dialect classification signal, and get direct ASR-provider status logs to confirm (rather than infer) the outage's root cause.
