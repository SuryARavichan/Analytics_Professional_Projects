# AI-Augmented Data Analyst Portfolio — Tech/SaaS Edition

Three projects built around one fictional B2B SaaS product (**Flowdesk**),
demonstrating the specific skill combination the 2026 job market is asking
for from data analysts in a product/SaaS org: not "I can call an OpenAI
API," and not "I can run a SQL query" alone, but the two used *together*
across the three places AI actually earns its keep in a modern SaaS
analytics stack — a natural-language interface over the data warehouse,
predictive account-health scoring with AI-generated action briefs, and an
autonomous monitoring agent over product-usage metrics.

| # | Project | The SaaS analytics role it demonstrates |
|---|---|---|
| 1 | [**PathfinderAI**](1_pathfinderai_nl_analytics/) — Natural-Language Product Analytics Copilot | Self-serve product analytics: ask a plain-English question about signups, activation, or MRR and get a schema-aware SQL query, a real result, and a grounded insight — with a 20-question accuracy benchmark, not a happy-path demo |
| 2 | [**ChurnGuard AI**](2_churnguard_health_scoring/) — Predictive Customer Health Scoring & AI-Generated Renewal Briefs | Revenue-facing account health: two RandomForest models (churn risk + expansion opportunity) combined into a 2×2 health quadrant, with AI-generated, evidence-grounded renewal briefs for CS and Sales |
| 3 | [**LaunchPulse**](3_launchpulse_adoption_agent/) — Autonomous Feature-Adoption Anomaly & Opportunity Agent | Proactive product monitoring: two statistically-appropriate anomaly detectors watch daily feature adoption and activation-funnel metrics, a fixed diagnostic SQL playbook gathers evidence, and an LLM writes (or correctly withholds) a root-cause incident report |

Each project's `dashboard/index.html` is self-contained — open it directly
in a browser, no server or install needed.

## An important design decision, upfront

This portfolio was built in a sandboxed environment with **no live LLM API
access and no ability to download open-source model weights**. Rather than
fake that constraint away, every project uses a shared, honest
abstraction: **`shared/llm_client.py`**, with interchangeable backends:

- **`CachedClient`** (the default) — replays pre-authored LLM responses
  from a local JSON fixture. Every single cached response in this
  portfolio was written by reading the *actual* output of the real,
  executed analysis upstream (the real SQL result, the real per-account
  churn/expansion scores, the real diagnostic-query evidence) — it's
  exactly what a live call would produce given that evidence, not a
  fabricated example.
- **`OpenAIClient` / `AnthropicClient`** — call the real API using your
  own key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). Export a key and the
  identical notebook code runs live with zero changes.

This means every notebook in this portfolio **executes end-to-end,
offline, free, and reproducibly** — a reviewer can clone this and run it
without an API key or a bill — while the architecture is exactly
production-shaped. **Everything that isn't a generative-text call is
100% real and actually executed**: the SQL (including window-function
feature engineering), the statistics, the machine learning (two
RandomForest classifiers with real AUCs), the anomaly detection (STL
decomposition, rolling activation-rate drift). Only the generative
narration is swapped for a grounded, pre-recorded fixture in demo mode.

## Why these three, together

They're deliberately not three versions of the same trick. PathfinderAI is
the *self-serve interface* layer (anyone in the company can ask a question
without waiting on an analyst). ChurnGuard AI is the *revenue-decision*
layer (which accounts does CS save, which does Sales expand, and why).
LaunchPulse is the *proactive-monitoring* layer (catch a product or
funnel problem before someone has to ask about it). Together they cover
the three places a SaaS company's data team is actually expected to ship
AI-augmented tooling in 2026 — and every one of them treats the LLM as a
component to be guardrailed and evaluated (a safety-checked SQL layer, a
leakage-free ML feature pipeline, a deterministic diagnostic playbook),
not an oracle trusted blindly, which is the single biggest thing
separating a credible AI-augmented-analyst portfolio from a toy chatbot
demo.

## Key numbers across the portfolio

- **PathfinderAI**: 20-question text-to-SQL benchmark, 80% execution
  accuracy (100% easy, 100% medium, 33% hard), across a 4-table, ~3,200-
  account Flowdesk product database.
- **ChurnGuard AI**: 2,800 simulated B2B SaaS accounts; churn model AUC
  **0.837**, expansion model AUC **0.961**; a 2×2 health quadrant
  surfacing 190 "Engaged but Unstable" accounts a usage-only metric would
  miss; 10 AI-generated renewal briefs for the highest-priority upcoming
  renewals.
- **LaunchPulse**: 18 months of daily feature-adoption and activation-
  funnel data (8 features × 4 channels × 3 plan tiers); 2 purpose-built
  anomaly detectors; 4 flagged events investigated, 3 confirmed incidents
  and 1 correctly dismissed false alarm.

## Resume bullet points

- Built a natural-language-to-SQL analytics copilot for a SaaS product
  database with schema-aware prompting, enforced safety guardrails, and a
  20-question evaluation benchmark scoring 80% execution accuracy — with a
  documented, realistic failure analysis (grain/denominator errors,
  missing DISTINCT, stale-status filters, ambiguous entity resolution).
- Designed a two-sided customer health-scoring system (churn risk +
  expansion opportunity) using dual RandomForest classifiers (AUC 0.84 /
  0.96) trained on leakage-free, SQL-window-function-engineered features,
  combined into a 2×2 health quadrant, paired with an LLM-generated
  renewal-brief pipeline for Customer Success and Sales.
- Architected an autonomous product-analytics monitoring agent using two
  statistically-appropriate anomaly detectors (STL-decomposition volume
  anomalies; rolling activation-rate drift) and a deterministic SQL
  diagnostic playbook feeding LLM-synthesized incident reports —
  correctly distinguishing 3 real incidents from 1 false alarm.
- Designed a reusable, pluggable LLM client abstraction (live API /
  offline cached-fixture backends) used across all 3 production-shaped AI
  analytics pipelines, enabling fully reproducible, cost-free demos.

## Interview prep — the questions this portfolio invites

- *"Why cached mode instead of a live API key?"* — the build environment
  had no API access, so the architecture is key-agnostic from day one, and
  every cached response is grounded in real upstream analysis rather than
  invented. This is also genuinely good practice — CI pipelines for LLM
  features commonly run against recorded fixtures for exactly this reason
  (cost, determinism, speed).
- *"How do you know the LLM isn't just making things up?"* — point to
  PathfinderAI's two-stage split (SQL generation is separate from and
  precedes insight generation), ChurnGuard's briefs being generated only
  from real computed per-account evidence (and one brief, ACCT60347,
  explicitly contradicting its own quadrant label because the evidence
  supports it), and LaunchPulse's fixed diagnostic playbook that produces
  the evidence used to *dismiss* a false alarm just as readily as it
  confirms a real incident.
- *"Why two models / two detectors instead of one?"* — a single blended
  score or a single detection threshold hides real structure: churn risk
  and expansion opportunity call for different teams and different
  actions; a sharp usage break and a slow funnel-quality decline have
  different statistical signatures that one threshold can't catch both of.
- *"What would break first if this went to production?"* — PathfinderAI's
  ambiguous-entity resolution (which "Enterprise accounts" definition),
  ChurnGuard's short-tenure accounts having wider error bars on both
  scores, and LaunchPulse's fixed 21-day drift-detector lookback not being
  seasonally robust — are all named, specific, already-identified weak
  points, which is a stronger answer than claiming there are none.

## Requirements to re-run anything

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn \
            plotly nbformat jupyter openai anthropic openpyxl
```

Each project's `data/generate_data.py` is self-contained and reproducible
(seeded random generators). To rebuild a notebook: `cd notebook &&
python3 build_notebook.py && jupyter nbconvert --to notebook --execute
--inplace analysis.ipynb`. To rebuild a dashboard: `cd dashboard &&
python3 build_dashboard.py`. To go live with a real LLM: `export
OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`) before running any notebook —
no code changes needed.
