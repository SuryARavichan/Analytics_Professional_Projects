# ResolveIQ — Autonomous CX Agent Performance & Containment Analytics

**One-line pitch:** an AgentOps analytics system for an autonomous AI customer-support-agent platform that catches the AI system's own self-reported success metric being gamed by a policy change — a +20.2pp "containment win" that turns out to be completely flat (p=0.98) once adjusted for tickets customers had to reopen.

## Why this project, and why it's not "just a support-ticket dashboard"

Agentic AI products don't just answer questions anymore — they take autonomous multi-step actions and decide for themselves when a task is "done." That creates a new measurement problem traditional support analytics never had: the system reporting on its own success can be wrong, and can be wrong in a way that looks like a big win. This project builds the specific check that catches it — validating an agent's official containment metric against what customers actually did afterward (reopened the ticket or not) — and finds exactly that failure mode hiding inside a release that looks like a clean success at the headline level.

## What's actually built here (not just described)

- **`data/generate_data.py`** — 18,000 support tickets for Ravel AI (a fictional autonomous CX agent platform) across 4 categories and a v1→v2 agent release, with realistic weekday-weighted volume, category-specific containment/reopen/cost dynamics, and a deliberately embedded "metric gaming" incident: account_recovery's v2 release lets the agent take autonomous account actions instead of always escalating, inflating official containment while reopen rate more than triples.
- **`pipeline/containment.py`** — computes both official (agent-self-reported) and true (reopen-within-48h-adjusted) containment rate, a two-proportion z-test usable on either metric, and an explicit 4-way rubric (`GENUINE_IMPROVEMENT` / `FALSE_IMPROVEMENT` / `REGRESSION` / `NO_CHANGE`) that only calls something a real win when *both* metrics agree.
- **`pipeline/economics.py`** — cost-per-ticket analysis that isolates whether a containment policy change actually saves money once the downstream cost of reopened tickets (many needing a human anyway) is counted.
- **`pipeline/agent.py`** — the perceive → act → reason → report loop: for each category, an LLM applies the classification rubric to both statistical tests plus the cost breakdown and writes a release-review note — explicitly required to flag a `FALSE_IMPROVEMENT` verdict as a metric-gaming risk rather than reporting the flattering official number.
- **`notebook/analysis.ipynb`** — the full, executed walkthrough: containment summary by category/version, the side-by-side statistical tests, the cost/reopen-rate trade-off, and all four agent release-review reports. Zero execution errors (18 cells).
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards, a 4-panel Plotly view, and all four category reports rendered as verdict-coded cards.

## Key results

| Category | Official lift | Official p-value | True lift | True p-value | Verdict |
|---|---|---|---|---|---|
| billing | +9.03pp | 4.9×10⁻¹⁵ | +7.80pp | 1.4×10⁻¹⁰ | **Genuine improvement** |
| technical | +6.75pp | 6.1×10⁻⁷ | +7.54pp | 5.1×10⁻⁸ | **Genuine improvement** |
| shipping | −1.26pp | 0.40 | −1.40pp | 0.36 | **No change (control)** |
| account_recovery | **+20.24pp** | <0.001 | **−0.04pp** | **0.98** | **False improvement** |

Account_recovery's reopen rate rose from 9.8% to 33.9% (v1→v2), and average cost per ticket stayed roughly flat ($3.89 → $3.65) — so the policy change delivered no cost benefit either, despite the misleadingly large official metric.

## On the honest limitations (stated in the notebook, not hidden)

- **The 48-hour reopen window is a proxy for "didn't actually work," not a perfect one.** Some reopens are unrelated follow-up questions; some genuine failures surface after 48 hours and are missed. The notebook states this explicitly — a production version would want a longer observation window and ideally a direct outcome-verification signal, not just reopen behavior.
- **The rubric's significance threshold (α=0.01) is a stated choice**, matching the stricter-than-0.05 standard used elsewhere in this session's work for anything that gates an automated report, not derived from a formal cost-of-error analysis.
- **Root cause (the specific account-action failures) is inferred from the pattern, not directly observed** — this analysis identifies *that* the policy is failing and roughly *how* (reopen-rate spike, isolated to one category, no cost offset), not the specific failed actions themselves, which would need agent-transcript-level investigation as a follow-up.

## Resume bullets

- Built an AgentOps analytics system for an autonomous AI support-agent platform that detected an AI system's own self-reported success metric (containment rate) being gamed by a policy change — a +20.2pp headline "win" (p<0.001) that was completely flat (−0.04pp, p=0.98) once adjusted for customer reopen behavior.
- Designed a 4-way release-classification rubric (genuine improvement / false improvement / regression / no change) requiring agreement between an AI system's self-reported outcome metric and an independently-validated true outcome metric before reporting any release as a win.
- Paired a statistical containment-quality finding with cost analysis, showing a "containment improvement" delivered zero cost benefit once the downstream cost of reopened tickets was counted — reframing a headline metric win as a net-neutral-to-negative outcome.

## Interview talking points

- **"Why not just trust the agent's own resolution flag?"** — because an agentic system that can autonomously decide a task is "done" can be wrong about that in a way that inflates its own scorecard; account_recovery is a direct demonstration of exactly that failure mode, caught only by checking real downstream customer behavior (did they come back?) against the agent's self-report.
- **"How is this different from a standard A/B test regression check?"** — a standard regression check (like the deploy-regression detector in the companion SentryOps AI project) asks "did the outcome metric get worse?" This asks a sharper question: "did the outcome metric's own definition become untrustworthy?" — the official metric here didn't get worse, it got better, and that's exactly what made it dangerous to trust at face value.
- **"What's the actual business risk if this ships unnoticed?"** — leadership sees a "+20pp containment" release note, allocates more autonomy/scope to the account_recovery agent, and customer trust erodes through repeated failed self-service attempts — a much worse outcome than if the release had simply shown no improvement.
- **"What would you change with more time/data?"** — extend the observation window past 48 hours and add a direct outcome-verification signal (e.g. confirming the account action succeeded via a follow-up system check) rather than relying on reopen behavior as a proxy for failure.
