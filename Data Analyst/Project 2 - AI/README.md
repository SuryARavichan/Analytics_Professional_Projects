# AI-Augmented Data Analyst Portfolio — 3 Projects

Three projects built to demonstrate the specific skill combination the
2026 job market is asking for from data analysts: not "I can call an
OpenAI API," and not "I can run a SQL query" alone, but the two used
*together*, with the AI layer properly guardrailed, evaluated, and
grounded in real analysis rather than trusted blindly.

| # | Project | The "AI + analytics" archetype it demonstrates |
|---|---|---|
| 1 | [**QueryMind**](1_querymind_nl_to_sql/) — Natural-Language-to-SQL Analytics Copilot | The conversational BI interface pattern (Cortex Analyst, Tableau Pulse, ThoughtSpot Sage) — with a real 20-question accuracy benchmark, not a happy-path demo |
| 2 | [**EchoIQ**](2_echoiq_voice_of_customer/) — AI Voice-of-Customer Intelligence Engine | Unstructured-data intelligence — sentiment validated against ground truth, unsupervised theme discovery, LLM summarization grounded in real clusters |
| 3 | [**SentinelBI**](3_sentinelbi_anomaly_agent/) — Autonomous Anomaly Detection & AI Incident-Reporting Agent | The agentic pattern (perceive → act → reason → report) — real statistical detection, real diagnostic SQL, an LLM that can say "this isn't real" |

Each project's `dashboard/index.html` is self-contained — open it directly
in a browser, no server or install needed.

## An important design decision, upfront

This portfolio was built in a sandboxed environment with **no live LLM API
access and no ability to download open-source model weights**. Rather than
fake that constraint away, every project uses a shared, honest
abstraction: **`shared/llm_client.py`**, with two backends:

- **`CachedClient`** (the default) — replays pre-authored LLM responses
  from a local JSON fixture. Every single cached response in this
  portfolio was written by reading the *actual* output of the real,
  executed analysis upstream (the real SQL result, the real cluster's top
  terms, the real anomaly diagnostics) — it's exactly what a live call
  would produce, not a fabricated example.
- **`LiveClient`** (`OpenAIClient` / `AnthropicClient`) — calls the real
  API using your own key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). Export
  a key and the identical notebook code runs live with zero changes.

This means every notebook in this portfolio **executes end-to-end,
offline, free, and reproducibly** — a reviewer can clone this and run it
without an API key or a bill — while the architecture is exactly
production-shaped. **Everything that isn't a generative-text call is
100% real and actually executed**: the SQL, the statistics (z-tests, STL
decomposition, power-style reasoning), the clustering (TF-IDF + KMeans),
the sentiment scoring (VADER), the anomaly detection. Only the generative
narration is swapped for a grounded, pre-recorded fixture in demo mode.

## Why these three, together

They're deliberately not three versions of the same trick. QueryMind is
the *interface* layer (turn a question into an answer). EchoIQ is the
*unstructured-data* layer (turn raw text into structured signal). SentinelBI
is the *monitoring/agentic* layer (turn a metric stream into a proactive,
investigated alert). Together they cover the three places "AI" actually
shows up in a modern analytics stack — and every one of them treats the
LLM as a component to be guardrailed and evaluated, not an oracle to be
trusted, which is the single biggest thing separating a credible
AI-augmented-analyst portfolio from a toy chatbot demo.

## Resume bullet points

- Built an LLM-powered natural-language-to-SQL analytics copilot with
  schema-aware prompting, enforced safety guardrails (blocking write/
  injection attempts), and a 20-question evaluation benchmark scoring
  80% execution accuracy — with a documented, realistic failure analysis
  (grain errors, missing DISTINCT, dropped filters, ambiguous entity
  resolution).
- Designed an unstructured Voice-of-Customer pipeline combining VADER
  sentiment (validated against ground-truth star ratings), unsupervised
  TF-IDF/KMeans theme discovery, and LLM-generated executive digests —
  detecting a real product regression from review text alone, confirmed
  independently against a second data source.
- Architected an autonomous anomaly-detection agent using two
  statistically-appropriate detection methods (STL decomposition +
  residual z-score; a custom regional-drift detector), a deterministic SQL
  diagnostic playbook, and LLM-synthesized root-cause incident reports —
  including correctly dismissing a false alarm rather than over-alerting.
- Designed a reusable, pluggable LLM client abstraction (live API /
  offline cached-fixture backends) used across 3 production-shaped AI
  analytics pipelines, enabling fully reproducible, cost-free demos.

## Interview prep — the questions this portfolio invites

- *"Why cached mode instead of a live API key?"* — Be straightforward:
  the build environment had no API access, so the architecture is built
  to be key-agnostic from day one, and every cached response is grounded
  in the real upstream analysis rather than invented. This is also
  genuinely good practice — CI pipelines for LLM features commonly run
  against recorded fixtures for exactly this reason (cost, determinism,
  speed).
- *"How do you know the LLM isn't just making things up?"* — Point to the
  two-stage split in QueryMind (SQL generation is separate from and
  precedes insight generation, so the narrative only ever sees a real
  query result), and to SentinelBI's fixed diagnostic playbook (the LLM
  synthesizes evidence gathered by real SQL, it doesn't investigate on its
  own authority).
- *"What would break first if this went to production?"* — The
  attribution-style ambiguity classes: QueryMind's ambiguous-column
  resolution (store region vs. customer region), EchoIQ's catch-all
  cluster needing a second sub-clustering pass, and SentinelBI's
  holiday-seasonality false-positive class are all named, specific,
  already-identified weak points — which is a stronger answer than
  claiming there are none.
- *"Why build the statistics/SQL layer yourself instead of just prompting
  an LLM to 'analyze this data'?"* — Determinism and auditability. A
  number a stakeholder will act on needs to trace back to a real query or
  a real statistical test, not a model's best guess at what the data
  probably says.

## Requirements to re-run anything

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn \
            plotly nbformat jupyter vaderSentiment openai anthropic
```

Each project's `data/generate_data.py` is self-contained and reproducible
(seeded random generators). To rebuild a notebook: `cd notebook &&
python3 build_notebook.py && jupyter nbconvert --to notebook --execute
--inplace analysis.ipynb`. To rebuild a dashboard: `cd dashboard &&
python3 build_dashboard.py`. To go live with a real LLM: `export
OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`) before running any notebook —
no code changes needed.
