# LaunchPulse — Autonomous Feature-Adoption Anomaly & Opportunity Agent

**One-line pitch:** an agent that continuously watches a SaaS product's daily feature-adoption and activation-funnel metrics, detects anomalies with two statistical methods built for different anomaly shapes, runs a deterministic diagnostic playbook the instant one fires, and has an LLM turn the gathered evidence into a root-cause incident report for a Product Manager — including correctly recognizing when a flagged spike isn't actually a problem.

## Why this project, and why "two detectors" matters

A broken feature release and a slow channel-quality decline look nothing alike statistically. One is a sharp, single/multi-day break in usage *volume* — the kind of thing a seasonal-decomposition + z-score detector is built for. The other is a gradual decline in a *rate* (signups converting to activated accounts) that barely moves total volume at all, because the channel is still sending the same number of signups — they just aren't converting. A single detector tuned to catch one shape will structurally miss the other, no matter how the threshold is adjusted. This project builds and validates both:

- **`point_anomaly_detector()`** — STL seasonal decomposition (7-day period, since Flowdesk usage is heavily weekday-skewed, a B2B pattern) + residual z-score. Catches sharp breaks in total daily adoption-event volume.
- **`channel_drift_detector()`** — compares each acquisition channel's trailing 7-day activation rate to its rate from 3 weeks earlier, flagging a sustained multi-day decline. Catches the funnel-quality problem the volume detector can't see.

## What's actually built here (not just described)

- **`data/generate_data.py`** — simulates 18 months of daily Flowdesk product usage (52,512 rows across 8 features × 4 acquisition channels × 3 plan tiers) plus a parallel channel-level signup/activation funnel (2,188 rows), with 4 injected, realistic events: a broken-release single-day feature outage, a 3-day feature-launch spike, a 3-week channel-specific activation-rate drift, and a broad marketing-webinar traffic bump deliberately designed to *look* like an anomaly but not be one.
- **`pipeline/anomaly_detector.py`** — both detectors, real statistics (`statsmodels` STL decomposition; rolling-window percent-change with run-length filtering), documented limitations included (a fixed volume threshold misses gradual drift; a fixed 21-day-lookback drift comparison isn't seasonally robust across a real holiday window — the same honest-limitations approach used in the companion SentinelBI project).
- **`pipeline/diagnostics.py`** — the deterministic diagnostic playbook: `breakdown_by_feature`, `breakdown_by_channel`, `compare_to_baseline`, and `channel_funnel_breakdown` — 4 real, parameterized SQL queries against the SQLite database, always run the same way for every incident.
- **`pipeline/agent.py`** — the perceive → act → reason → report orchestration (`investigate()`), with an explicit scoping note that this is a fixed playbook, not a dynamic tool-selection loop, and why that's the right v1 choice.
- **`notebook/analysis.ipynb`** — the full, executed analysis: both detectors run and visualized, all 4 incidents investigated end-to-end, and a 5-point executive summary. Zero execution errors.
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards, an interactive Plotly timeline marking every flagged event (color-coded confirmed vs. dismissed), and all 4 agent incident reports rendered as cards with their full supporting evidence tables.

## The 4 incidents (and why one of them should be dismissed, not escalated)

| Incident | Type | Verdict | Key evidence |
|---|---|---|---|
| Automation Rules outage (Oct 8, 2024) | Point anomaly | **Confirmed** | That feature dropped 86.5% vs. baseline; every other feature moved +6% to +13% the same day |
| AI Assistant launch spike (Apr 14-16, 2025) | Point anomaly | **Confirmed** | That feature rose 304% vs. baseline; every other feature rose only 13-20% in the same window |
| Paid Social activation drift (Dec 14-24, 2024) | Channel drift | **Confirmed** | Activation rate fell 16.5 points on that channel alone; every other channel held flat |
| Feb 11-13, 2025 broad spike | Point anomaly | **Dismissed — false alarm** | Every feature rose 91-102%, and channel mix stayed proportional to normal traffic share — the signature of a known, planned marketing webinar, not a localized problem |

That last row is the finding worth calling out in an interview: an agent that manufactures a root-cause story for every flagged spike isn't trustworthy. The same diagnostic playbook that isolates a real single-feature outage also produces the *uniform* evidence pattern that correctly rules a broad marketing event out as a non-incident — the dismissal capability is as important as the diagnosis capability.

## On the LLM backend

This sandbox has no live LLM API access. `shared/llm_client.py` implements a small `LLMClient` abstraction with interchangeable backends — `CachedClient` (replays real, pre-authored JSON responses, what this build uses) and `OpenAIClient`/`AnthropicClient` (drop-in live backends, auto-selected by `get_client()` when an API key is present). Every cached incident report was authored by reading the *actual* diagnostic query results this notebook computes — the same evidence a live model call would receive — not invented.

## Resume bullets

- Built an autonomous product-analytics monitoring agent combining two purpose-built statistical anomaly detectors (STL-decomposition volume anomalies and rolling activation-rate drift) to catch two structurally different failure shapes that a single detector would miss.
- Designed a perceive-act-reason-report agent architecture where a fixed, auditable SQL diagnostic playbook gathers evidence before an LLM synthesizes a root-cause incident report, correctly distinguishing 3 real incidents from 1 false alarm using the same evidence-gathering pipeline.
- Simulated and validated (via injected, statistically-confirmed events) an 18-month, 8-feature × 4-channel product-usage dataset, with all four detector outputs matching the intended incident windows before any downstream analysis was built on top.

## Interview talking points

- **"Why two detectors instead of tuning one threshold?"** — a broken-release outage and a channel activation-rate drift have fundamentally different statistical signatures (a break in a level vs. a decline in a rate); no single threshold catches both without either missing the slow one or false-alarming on daily noise.
- **"How do you know the agent isn't just rationalizing whatever the detector flagged?"** — the Feb 11-13 case: the same 4-query diagnostic playbook that isolated the single-feature Automation Rules outage also produced clean evidence (uniform lift across every feature and channel, proportional to normal traffic mix) that the model used to correctly conclude no action was needed.
- **"What's the biggest limitation you'd flag yourself?"** — the drift detector's 21-day-lookback baseline isn't seasonally robust across genuinely elevated periods (documented in the module docstring, and handled via an `exclude_windows` parameter, the same mitigation used in the companion SentinelBI project); a production version would compare against a seasonally-adjusted trend instead of a raw N-days-ago snapshot.
- **"What would you build next?"** — letting the agent choose which diagnostic query to run next (a real ReAct-style tool-selection loop) for incidents the fixed playbook doesn't localize on the first pass, and extending the channel-drift detector to plan-tier and feature-level granularity, not just acquisition channel.
