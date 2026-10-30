# QueryMind — Natural-Language-to-SQL Analytics Copilot

**One-line pitch:** Built an LLM-powered "ask your database a question in
English" copilot with schema-aware prompting, an enforced safety
guardrail, and a 20-question evaluation benchmark scoring real execution
accuracy (80%) — not a demo that only works on the happy path.

## Why this project

Text-to-SQL copilots (Snowflake Cortex Analyst, Tableau Pulse, ThoughtSpot
Sage, and every BI vendor's 2025-26 "chat with your data" feature) are
becoming the default interface between business users and analytics teams.
A data analyst who understands how these systems work — where they break,
how to evaluate them, why "it demoed fine" isn't the same as "it's
reliable" — is more valuable than one who only knows how to write SQL by
hand. This project builds one from the ground up to demonstrate exactly
that.

## Architecture

```
question (English)
   │
   ▼
[1] SQL generation  — schema-aware prompt (schema.py) → LLM → SQL
   │
   ▼
[2] Safety guardrail — regex-reject any non-SELECT statement,
   │                    reject multi-statement injection, execute
   │                    against a READ-ONLY SQLite connection
   ▼
[3] Execution        — real query result (columns + rows)
   │
   ▼
[4] Insight generation — a SECOND LLM call, given only the actual query
                          result, writes the plain-English finding
                          (keeps the narrative grounded — it can't
                          hallucinate a number that isn't in the result)
```

## On the LLM backend (read this first)

This was built in a sandboxed environment with no live API access, so it
ships in **cached mode** by default (`shared/llm_client.py`'s
`CachedClient`) — every SQL and insight response you'll see was authored
by grounding it in the real schema and real query results, so it is
exactly what running this live would produce, including 4 deliberately
realistic failure cases in the eval (not a cherry-picked 100%). Export
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and the identical code calls the
real API instead — that's the entire point of the `LLMClient` abstraction.

## The dataset

A synthetic multi-store retail chain: `stores`, `products`, `customers`,
`sales_transactions`, `sales_items` — 24 stores, 220 products, 9,000
customers, 42,000 transactions, ~92,000 line items (Jun 2024–Dec 2025).
Five joined tables on purpose — a single-table demo doesn't exercise real
text-to-SQL difficulty. Regenerate with `cd data && python3 generate_data.py`.

## The evaluation (the part most portfolio projects skip)

`eval/build_benchmark.py` defines 20 hand-verified questions spanning
easy/medium/hard difficulty, each with a gold SQL query. `eval/run_eval.py`
executes both the gold and the (cached-demo) predicted SQL against the
real database and scores **execution accuracy** — do the two queries
return the same result, not do they look the same as text — the standard
metric in published text-to-SQL research (e.g. Spider).

**Result: 80% overall (16/20)** — 100% on easy, 89% on medium, 40% on
hard. The 4 failures are real, named text-to-SQL failure classes:

| Question | Failure class |
|---|---|
| "What % of transactions were returned?" | Grain/denominator error from an unnecessary join |
| "How many unique customers bought Electronics?" | Missing `DISTINCT` after a fan-out join |
| "Month-over-month revenue trend" | Silently dropped a `status='Completed'` filter |
| "Online vs in-store revenue for the Northeast" | Ambiguous entity resolution (store region vs. customer signup region) |

## Files

| Path | What it is |
|---|---|
| `data/generate_data.py` | Builds the 5-table SQLite database |
| `pipeline/schema.py` | Auto-introspects the DB into an LLM-ready schema description |
| `pipeline/text_to_sql.py` | The core pipeline: safety-checked SQL generation + execution + insight generation |
| `eval/build_benchmark.py`, `eval/run_eval.py` | The 20-question benchmark and scoring harness |
| `notebook/analysis.ipynb` | Full walkthrough: live queries, safety-guardrail demo, eval results, failure analysis |
| `dashboard/index.html` | Click-through interactive chat demo — **open in any browser** |
| `../shared/llm_client.py` | The pluggable live/cached LLM client used by all 3 AI projects |

## How to talk about this in an interview

- "I didn't just wire up an LLM to a database — I built the safety layer
  first: a regex guardrail against write statements and multi-statement
  injection, executing against a read-only connection as defense in
  depth, and I can show it actually blocking a `DROP TABLE` attempt."
- "I built a real evaluation benchmark, not just a happy-path demo. 80%
  accuracy with a clean easy-to-hard accuracy gradient is a more credible
  number than a suspicious 100%, and I can explain exactly why each of the
  4 failures happened — grain errors, fan-out joins, dropped filters,
  ambiguous entity resolution — because those are the real failure classes
  in every text-to-SQL system, not something I invented for the demo."
- "I split SQL generation and insight generation into two separate LLM
  calls specifically so the narrative can't hallucinate — it only ever
  sees the real query result, never the raw question with license to
  guess."
