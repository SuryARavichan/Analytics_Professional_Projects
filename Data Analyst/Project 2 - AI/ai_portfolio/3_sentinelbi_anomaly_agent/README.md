# SentinelBI — Autonomous Anomaly Detection & AI Incident-Reporting Agent

**One-line pitch:** Built an agent that detects revenue anomalies with two
statistically-appropriate methods, automatically gathers real diagnostic
evidence via SQL, and has an LLM write an auditable root-cause incident
report — including correctly recognizing and dismissing a false alarm
instead of escalating everything.

## Why this project

"Agentic AI" is 2026's single most-searched analytics buzzword, and most
of what gets built under that name is a thin LLM wrapper with no real
statistical or diagnostic substance underneath. This project is the
opposite: the detection is real applied statistics (STL decomposition, a
custom drift detector), the diagnosis is real SQL that runs before the LLM
says anything, and the LLM's role is narrowly scoped to what LLMs are
actually good at — synthesizing already-gathered evidence into a clear,
readable narrative, not being trusted to invent the analysis itself.

## Architecture: perceive → act → reason → report

```
Daily revenue data (region x channel)
   │
   ▼
[PERCEIVE] Two detectors, chosen for anomaly SHAPE:
   │  • point_anomaly_detector — STL decomposition + residual z-score
   │    (catches sharp single/multi-day spikes and drops)
   │  • regional_drift_detector — trailing 7-day avg vs. 3-weeks-prior,
   │    per region (catches slow decline invisible in the company total)
   ▼
[ACT] Fixed diagnostic playbook (diagnostics.py) — real SQL:
   │  breakdown by region, breakdown by channel, comparison vs. baseline
   ▼
[REASON] LLM synthesizes the gathered evidence into root cause + confidence + action
   ▼
[REPORT] Structured incident report, rendered in the dashboard timeline
```

## On the LLM backend

Same design as QueryMind and EchoIQ — see `../shared/llm_client.py`. No
live API access in this sandbox, so every incident report runs in
`CachedClient` mode by default, authored by reading the *actual*
diagnostic query results computed in this repo — not invented. Export an
API key and the identical code calls a real model with the same evidence.

## The dataset & injected incidents

18 months (Jan 2024–Jun 2025) of daily revenue for "NimbusCart," a
fictional e-commerce retailer, broken out by 4 regions × 3 channels.
Three distinct incidents are injected, deliberately of different shapes:

1. **Payment-gateway outage** (Sep 12, 2024) — single day, isolated to
   the Web channel (-77% vs. baseline; other channels unaffected).
2. **Regional promo spike** (Nov 29–Dec 1, 2024) — West region up +240%
   vs. baseline during Black Friday/Cyber Monday, while other regions saw
   a normal +27-37% seasonal lift.
3. **Slow regional decline** (Feb 1-21, 2025) — South region down -16.8%
   over 3 weeks while every other region was flat-to-up. **Total company
   revenue moved only -1.2%** over this window — invisible at the
   aggregate level, which is exactly why this project uses two detectors
   instead of one.

Regenerate with `cd data && python3 generate_data.py`.

## The part that makes this credible, not just a demo

- **Two detectors because one anomaly shape doesn't fit all.** The South
  regional decline would never have crossed a company-total z-score
  threshold; a region-level drift check was necessary to catch it at all.
- **A documented detector limitation, not a hidden one.** The regional-
  drift detector's fixed 21-day lookback isn't seasonally robust across
  the November-December holiday peak/trough — see the exclusion window
  and the docstring explaining exactly why, in `pipeline/anomaly_detector.py`.
- **The agent dismisses a false alarm.** A Dec 22-24 flag investigated by
  the same playbook shows uniform movement across every region and
  channel — the signature of a broad seasonal effect, not an isolated
  incident — and the report says so, with a downgraded confidence, instead
  of escalating every statistical flag as a "real" problem.
- **Diagnosis is real SQL, always the same 3 steps, before the LLM sees
  anything** — auditable, reproducible, and not dependent on the model
  choosing the right query on its own (a deliberate scope decision over a
  dynamic tool-selection loop — see `pipeline/agent.py`'s docstring for
  the tradeoff and the natural v2 extension).

## Recommendations delivered to "leadership"

1. Add a same-day alert on any single-channel revenue drop >50% vs.
   trailing baseline (would have caught the Sep 12 outage within hours).
2. Confirm and replicate the West-region promo mechanism that drove a
   240% lift, well above general seasonal effects.
3. Escalate the South-region decline to the regional commercial team, and
   add per-region trend monitoring as a standing dashboard view — this
   incident was invisible in the company-total view for three full weeks.
4. Add a holiday-calendar feature to the point-anomaly detector to stop
   the recurring Dec-ramp false-positive class.

## How to talk about this in an interview

- "I used two different detectors on purpose, not one tuned to catch
  everything — I can explain exactly why a company-total anomaly check
  would have missed the regional decline entirely."
- "The diagnostic evidence is real SQL that runs deterministically before
  the LLM does anything — the agent's report is auditable against actual
  query results, not just plausible-sounding text."
- "I built in a 'this isn't real' outcome on purpose. An agent that only
  ever escalates isn't trustworthy — I can show the exact evidence pattern
  (uniform movement across every region and channel) that tells it this
  one's a false alarm, not an incident."
- "I can point to a documented limitation in my own detector — the
  holiday-window exclusion — rather than pretending the system is perfect
  everywhere, and describe the real production fix (seasonally-adjusted
  baseline instead of fixed N-days-prior)."
