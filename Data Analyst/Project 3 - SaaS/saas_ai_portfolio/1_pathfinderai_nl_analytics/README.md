# PathfinderAI — Natural-Language Product Analytics Copilot for SaaS Growth Teams

**One-line pitch:** Built an LLM-powered "ask your funnel a question in
English" copilot purpose-fit to PLG/SaaS analytics (activation,
conversion, MRR, feature adoption, churn), with an enforced safety
guardrail and a 20-question benchmark scoring 80% execution accuracy —
including a failure case that's uniquely a SaaS terminology trap.

## Why this project

Every SaaS company's product/growth team asks some version of "what's our
activation rate," "which channel converts best," "which features are
sticky" dozens of times a week, usually by pinging a data analyst. A
copilot that answers these directly — safely, and with a track record you
can point to — is the exact interface pattern PLG companies are shipping
internally in 2026, and understanding how to build and evaluate one is a
sharper signal than a general "I used an LLM" project.

## Architecture

Same 4-stage pipeline as its sibling project QueryMind (schema-aware
prompt → safety-guarded execution → real query result → grounded insight
generation) — see `pipeline/text_to_sql.py`. What's different here is the
domain: a genuine PLG funnel schema (`accounts`, `users`, `subscriptions`,
`product_events`) instead of a retail schema, and an eval benchmark built
specifically around SaaS metrics and SaaS-specific ambiguity traps.

## On the LLM backend

No live API access in this sandbox, so it ships in `CachedClient` mode by
default (`shared/llm_client.py`) — every response was authored by
grounding it in the real schema and real query results. Export
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and the identical code goes live.

## The dataset

A synthetic PLG SaaS product ("Flowdesk," a team-collaboration app):
3,200 accounts, ~30,000 users, 3,200 subscription records, ~36,000
product-usage events, Apr 2024–Dec 2025. The funnel is simulated
end-to-end: signup → trial → activation (56% of signups) → paid
conversion (~21% blended) → expansion or churn, with realistic
acquisition-channel quality differences baked in (Referral converts at
30.4% vs. Paid Search at 11.4%). Regenerate with `cd data && python3
generate_data.py`.

## The evaluation

20 hand-verified questions across easy/medium/hard difficulty, scored on
**execution accuracy** (same methodology as QueryMind) — **80% overall
(16/20)**, 100% on easy and medium, 33% on hard. All 4 failures are
concentrated in the hard tier:

| Question | Failure class |
|---|---|
| "Unique accounts using a feature?" | Missing `DISTINCT` after a fan-out join on event logs |
| "Total current MRR by plan?" | Stale-data filter (didn't exclude Churned accounts whose `mrr` column is still populated) |
| "How many Enterprise accounts?" | **SaaS-specific ambiguity**: "Enterprise" as pricing plan vs. "Enterprise" as company size — ~1000+ employees ≠ Enterprise-plan subscribers |
| "% of activated accounts that convert?" | Grain/denominator error — joined the raw multi-row event log instead of a deduplicated activated-accounts set |

The "Enterprise accounts" ambiguity is worth calling out specifically: it
wouldn't occur in a non-SaaS schema, and it's exactly the kind of
domain-specific trap a generic, off-the-shelf benchmark wouldn't catch —
which is the argument for building an evaluation set tailored to your own
schema and terminology before shipping an internal copilot like this.

## How to talk about this in an interview

- "I reused the same guardrailed architecture as a prior project but
  built a fresh, domain-specific evaluation set — because a copilot that
  scores well on retail questions tells you nothing about whether it's
  safe to ship for a SaaS team's actual vocabulary and metrics."
- "The 'Enterprise accounts' failure is my favorite case in this whole
  portfolio — it's not a bug, it's a genuine ambiguity in business
  language, and the fix isn't a smarter model, it's a copilot that asks a
  clarifying question or states its interpretation instead of silently
  picking one."
- "I can walk through the funnel model end to end — why activation is
  defined the way it is, why the conversion-by-channel gap matters for
  budget allocation, why annual-billing share is a churn lever worth
  flagging."
