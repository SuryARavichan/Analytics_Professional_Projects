# EchoIQ — AI Voice-of-Customer Intelligence Engine

**One-line pitch:** Turned 5,200 unstructured app reviews and 1,400
support tickets into validated sentiment, unsupervised theme discovery,
and an AI-generated weekly digest — catching a shipped product regression
from text alone, confirmed independently across two data sources.

## Why this project

Every product/growth team sits on a pile of unstructured customer text
(reviews, tickets, NPS comments, chat transcripts) that mostly goes
unread. Turning that into structured, trend-able, alertable business
intelligence — without a human reading every entry — is one of the most
concretely valuable "AI + analytics" applications a company can point to
in 2026, and it's a skill combination (classical NLP + unsupervised ML +
LLM summarization, used together, with the LLM output *validated* rather
than trusted blindly) that's rarer than either "I can call an LLM API" or
"I can run KMeans" alone.

## Pipeline

```
5,200 reviews + 1,400 tickets
   │
   ▼
[1] Sentiment scoring — VADER (real, offline, lexicon-based)
   │
   ▼
[2] Validation — cross-check sentiment against star rating BEFORE
   │              trusting it for trend analysis (most portfolio NLP
   │              projects skip this step entirely)
   ▼
[3] Theme discovery — TF-IDF + KMeans, fully unsupervised (no predefined
   │                   keyword tags) — 8 themes emerge from the text itself
   ▼
[4] LLM cluster labeling — a human-readable label + context sentence per
   │                        cluster, grounded in its real top terms
   ▼
[5] Trend analysis + AI-generated weekly digest — surfaces a real incident
```

## On the LLM backend

Same design as QueryMind — see `../shared/llm_client.py`. This sandbox has
no live LLM API access, so cluster labels and the weekly digest run in
`CachedClient` mode by default, but every one of them was authored by
reading the *actual* top TF-IDF terms and *actual* monthly statistics
produced by the real, executed clustering and trend analysis — they are
exactly what a live GPT-4o-mini or Claude call would produce given this
data. Export an API key and the identical code calls the real model.

## The dataset

Synthetic but structurally realistic: 5,200 App Store/Play Store-style
reviews and 1,400 support tickets for "PulseFit," a fictional fitness
app, Nov 2024–Dec 2025. Text is built from theme+sentiment-conditioned
phrase banks (not an LLM) so the corpus has genuine discoverable thematic
structure. A deliberate ~12% star-rating/sentiment "surprise" rate is
baked in — real reviews are messier than a clean 1-star-is-always-negative
mapping, and that messiness is exactly what makes Section 1's validation
step meaningful rather than a formality. Regenerate with `cd data &&
python3 generate_data.py`.

## Key findings

- **Sentiment-vs-rating validation**: VADER agrees with the obvious
  direction of star rating for the large majority of reviews; the ~21.6%
  of 1-2-star reviews scoring "positive" and ~3.0% of 4-5-star reviews
  scoring "negative" are mostly genuinely mixed reviews (a compliment
  buried in a complaint, or vice versa) — worth knowing before trusting
  the sentiment trend downstream.
- **8 themes discovered with zero manual tagging**, cleanly separating a
  severe "App Stability Crisis" cluster (avg rating 2.28, crashes/broken
  updates) from a range of milder, mostly positive themes.
- **A shipped regression is directly visible in the data**: average
  rating drops from a 3.3-3.5 baseline to 2.5-2.6 in Jul-Aug 2025, and the
  App-Stability-Crisis cluster's share of reviews nearly doubles (14% →
  34%) in the exact same window, recovering by September — and the same
  pattern shows up **independently** in support-ticket volume and
  resolution rate over that period.

## Recommendations delivered to "leadership"

1. Wire the App-Stability-Crisis cluster's *share of incoming reviews*
   (not just average star rating, which lags and blends signal from every
   theme) into a release-health dashboard with an alert threshold — this
   would have flagged the July regression within days, not after a full
   rating dip was already visible externally.
2. Run a post-mortem on the July release to identify the testing gap.
3. The "General Dissatisfaction" catch-all cluster (27% of reviews, avg
   rating 2.44) is too broad to act on directly — recommend a second-pass
   sub-clustering restricted to just that cluster to split it into
   actionable sub-themes (pricing vs. missing-features vs. data-loss are
   likely three separate fixes, not one).

## How to talk about this in an interview

- "I validated the sentiment model against ground truth (star rating)
  *before* trusting it for anything — and I can point to the specific
  disagreement rate and explain why those cases are genuinely ambiguous
  reviews, not model failures."
- "The theme discovery is fully unsupervised — I didn't hand-tag 'crashes'
  as a category and go looking for it; the clustering surfaced it, and I
  only assigned the label afterward by reading the cluster's own top
  terms."
- "I confirmed the incident finding against a second, independent data
  source (support tickets) before presenting it as a real finding — single-
  source anomalies deserve a second look before you act on them."
- "The LLM's job here is summarization and labeling, grounded in real
  statistics I computed — it never invents the finding, it explains one
  that's already been measured."
