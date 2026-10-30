# CausalLift — Statistically Rigorous A/B Testing & Causal Impact Engine

**One-line pitch:** three real experimentation failure modes a senior product analyst is expected to catch — underpowered analysis that ignores available variance-reduction techniques, the "peeking problem" that quietly inflates false positives, and treating a non-randomized rollout like a clean A/B test — each demonstrated, quantified, and fixed on Flowdesk (the same fictional B2B SaaS product used elsewhere in this portfolio).

## Why this project, and why it's not "just another A/B test calculator"

Most portfolio A/B testing projects stop at a two-proportion z-test on a happy-path dataset. That demonstrates knowing the test exists, not knowing where the test breaks in practice. This project is built around three specific, well-documented ways real experimentation programs go wrong, and the standard industry fixes for each:

1. **Ignoring available variance reduction.** A pre-experiment covariate (e.g., a user's engagement level before they ever saw the treatment) is often available and correlated with the outcome — using it (CUPED) lets the same experiment detect the same effect with meaningfully more statistical power, for free.
2. **Checking the p-value every day and stopping as soon as it's significant.** This is one of the most common mistakes in applied experimentation, and it doesn't just slightly inflate the false-positive rate — this project's own Monte Carlo simulation shows it roughly **5x's** it (5% nominal → ~25% actual with 21 daily looks).
3. **Running a before/after comparison on a rollout that wasn't randomized** and calling the result causal. A staged, region-by-region rollout needs difference-in-differences (or a comparable design) — and a DiD estimate without a parallel-trends check and a placebo test isn't a trustworthy causal claim, just a plausible-looking number.

## What's actually built here (not just described)

- **`data/generate_data.py`** — three independent, purpose-built datasets: a randomized 12,000-user onboarding A/B test (with a pre-experiment covariate for CUPED), 21 days of daily conversion data for a pricing test with a **true null effect** (specifically seeded so naive daily peeking produces a real, undoctored false positive on day 8 — found by searching, not manufactured), and a 4-region × 16-week panel for a staged feature rollout with a true +5pp causal effect.
- **`pipeline/ab_testing.py`** — a two-proportion z-test, and a from-scratch CUPED implementation (`theta = Cov(y,x)/Var(x)` fit on the pooled sample, so it can't leak treatment-assignment information), plus a required-sample-size calculator used to sanity-check the experiment's power.
- **`pipeline/sequential_testing.py`** — `simulate_naive_peeking()` (Monte Carlo estimate of the true false-positive rate under repeated daily looks) and `calibrate_sequential_boundary()` (a bisection search that Monte-Carlo-calibrates an O'Brien-Fleming-shaped boundary schedule to hit a target 5% true error rate) — a legitimate, simulation-verified way to build a sequential testing boundary rather than citing a textbook table.
- **`pipeline/causal_inference.py`** — a two-way fixed-effects difference-in-differences estimator (`statsmodels` OLS with region and week fixed effects, cluster-robust SEs), plus `parallel_trends_check()` and `placebo_test()` — treated as first-class deliverables, not an afterthought, because a DiD estimate is not trustworthy without them.
- **`notebook/analysis.ipynb`** — the full, executed walkthrough of all three parts, including every chart. Zero execution errors.
- **`dashboard/index.html`** — self-contained interactive dashboard: KPI cards and a 4-panel Plotly view (the peeking demonstration on real data, the CUPED variance-reduction bar chart, the rollout region trends, and the naive-vs-calibrated false-positive-rate comparison).

## Key results

| Analysis | Result |
|---|---|
| Onboarding A/B test (Exp A) | +5.5pp activation (control 51.2% → treatment 56.7%), z=6.09, p<0.001 |
| CUPED on secondary metric | 44.4% variance reduction; t-statistic magnitude increased ~1.3x for the identical estimated effect |
| Naive daily-peeking false-positive rate | 24.9% actual vs. 5.0% nominal (5,000-simulation Monte Carlo, K=21 daily looks) |
| Calibrated sequential boundary | Achieves 5.1% true false-positive rate; correctly withholds significance on a real seeded instance where naive peeking would have flagged day 8 |
| Staged rollout (DiD) | +4.45pp causal effect (95% CI: 4.18pp to 4.72pp), p<0.001; clean placebo test (fake week-5 "rollout": +0.03pp, p=0.96) |

## On the honest limitations (stated in the notebook, not hidden)

- The DiD's cluster-robust standard errors are clustered by region, but with only 4 regions, the analytic cluster-robust formula's asymptotics are on shaky ground (statsmodels itself flags this). The standard fix at this cluster count is a **wild cluster bootstrap** (Cameron, Gelbach & Miller, 2008) — the notebook says so explicitly rather than reporting the p-value as exact.
- The sequential boundary shape used (`c * sqrt(K/k)`) is calibrated by simulation for this specific experiment design (K, n/day, baseline rate); a new experiment design would need re-calibration, which the module is built to do on demand rather than reusing a fixed table.

## Resume bullets

- Quantified the false-positive-rate inflation from naive daily-peeking in A/B testing via a 5,000-simulation Monte Carlo study (5% nominal → ~25% true rate with 21 daily looks) and implemented a Monte-Carlo-calibrated sequential testing boundary that controls it, verified against a real seeded instance of the failure mode.
- Implemented CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction from scratch, achieving a 44% reduction in outcome variance and demonstrating the same effect size detected with substantially more statistical power.
- Built a difference-in-differences causal-impact pipeline (two-way fixed effects, cluster-robust SEs) for a non-randomized staged feature rollout, including a parallel-trends check and placebo test, correctly identifying a +4.45pp causal effect while explicitly documenting the small-cluster standard-error limitation.

## Interview talking points

- **"Why does CUPED work?"** — it's regression adjustment / ANCOVA under the hood: if a pre-experiment covariate predicts the outcome and is uncorrelated with treatment (guaranteed by randomization), subtracting the covariate-explained portion of the outcome removes noise without introducing bias, because theta is fit on the pooled sample rather than per-arm.
- **"How do you know the calibrated boundary actually controls the error rate, and isn't just 'more conservative'?"** — it's calibrated via bisection search against a target false-positive rate (5%), verified by an independent Monte Carlo run (achieved 5.1%) — a boundary that's simply "more conservative" without calibration would under- or over-shoot the target, losing power or not actually fixing the problem.
- **"Why not just use a bigger sample and skip DiD's assumptions?"** — the rollout physically could not be randomized (it shipped by region on an operational timeline), which is the exact situation DiD exists for; the parallel-trends check and placebo test are what make the resulting causal claim defensible rather than just a plausible-looking correlation.
- **"What would you change with more time/data?"** — a wild cluster bootstrap for the DiD's standard errors given the small cluster count, and extending the sequential-testing module to support a continuous (not just binary) primary metric via a t-based rather than z-based boundary.
