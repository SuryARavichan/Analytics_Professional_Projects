"""
Generates 5 independent, realistic Flowdesk experiment "results packages"
for ShipCall AI, each purpose-built to require a DIFFERENT decision from
the agent's Ship / Hold / Iterate / Kill taxonomy -- because an agent that
always recommends "ship it" isn't making decisions, it's cheerleading.

  1. onboarding_redesign_v2   -> SHIP:   clean, significant primary lift,
                                  clean guardrails, no meaningful
                                  segment-effect heterogeneity.
  2. aggressive_upsell_banner -> HOLD:   primary metric IS significantly
                                  positive, but a guardrail metric
                                  (7-day churn) moves significantly worse
                                  -- a real net-negative-for-the-business
                                  risk a primary-metric-only view would miss.
  3. new_pricing_page_layout  -> ITERATE: overall effect is ~flat, but
                                  there's a real, significant heterogeneous
                                  treatment effect: strongly positive for
                                  Enterprise, negative for Starter -- the
                                  average hides a segment-specific win.
  4. search_ranking_algo_v3   -> INVALID (Sample Ratio Mismatch): the
                                  actual traffic split is significantly
                                  off from the intended 50/50 (a bucketing
                                  bug), so NEITHER the seemingly-positive
                                  primary result NOR anything else about
                                  this experiment can be trusted as-is.
  5. dashboard_load_speed_opt -> KILL:   no meaningful effect on the
                                  primary metric, clean guardrails, no
                                  HTE -- genuinely nothing to ship.

Each experiment includes: overall primary-metric counts, 2 guardrail
metrics (with a "lower_is_better" direction flag), and a 3-segment
(Starter/Growth/Enterprise) breakdown for heterogeneous-treatment-effect
(HTE) testing.
"""
import json
import numpy as np

rng = np.random.default_rng(909)


def binom(n, p):
    return int(rng.binomial(n, p))


experiments = []

# ============================================================
# 1. Onboarding Redesign v2 -- SHIP
# ============================================================
seg_defs = [("Starter", 2500, 0.40, 0.445), ("Growth", 1750, 0.44, 0.485), ("Enterprise", 750, 0.50, 0.545)]
segments = [{"segment": s, "control_n": n, "control_conversions": binom(n, pc),
             "treatment_n": n, "treatment_conversions": binom(n, pt)} for s, n, pc, pt in seg_defs]
control_n = sum(s["control_n"] for s in segments)
treatment_n = sum(s["treatment_n"] for s in segments)
experiments.append({
    "experiment_id": "onboarding_redesign_v2", "name": "Onboarding Redesign v2",
    "primary_metric": "activation_rate", "hypothesis": "Simplifying the setup wizard from 6 steps to 3 increases activation.",
    "control_n": control_n, "control_conversions": sum(s["control_conversions"] for s in segments),
    "treatment_n": treatment_n, "treatment_conversions": sum(s["treatment_conversions"] for s in segments),
    "guardrails": [
        {"name": "day7_churn_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.085),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.080)},
        {"name": "support_tickets_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.120),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.115)},
    ],
    "segments": segments, "intended_split": 0.5,
})

# ============================================================
# 2. Aggressive Upsell Banner -- HOLD (guardrail violation)
# ============================================================
seg_defs = [("Starter", 3000, 0.16, 0.185), ("Growth", 2000, 0.19, 0.215), ("Enterprise", 1000, 0.22, 0.245)]
segments = [{"segment": s, "control_n": n, "control_conversions": binom(n, pc),
             "treatment_n": n, "treatment_conversions": binom(n, pt)} for s, n, pc, pt in seg_defs]
control_n = sum(s["control_n"] for s in segments)
treatment_n = sum(s["treatment_n"] for s in segments)
experiments.append({
    "experiment_id": "aggressive_upsell_banner", "name": "Aggressive Upsell Banner",
    "primary_metric": "trial_to_paid_conversion", "hypothesis": "A persistent in-app upgrade banner increases trial-to-paid conversion.",
    "control_n": control_n, "control_conversions": sum(s["control_conversions"] for s in segments),
    "treatment_n": treatment_n, "treatment_conversions": sum(s["treatment_conversions"] for s in segments),
    "guardrails": [
        {"name": "day7_churn_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.090),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.117)},
        {"name": "support_tickets_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.100),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.106)},
    ],
    "segments": segments, "intended_split": 0.5,
})

# ============================================================
# 3. New Pricing Page Layout -- ITERATE (heterogeneous treatment effect)
# ============================================================
seg_defs = [("Starter", 3000, 0.20, 0.17), ("Growth", 2000, 0.23, 0.235), ("Enterprise", 1000, 0.28, 0.36)]
segments = [{"segment": s, "control_n": n, "control_conversions": binom(n, pc),
             "treatment_n": n, "treatment_conversions": binom(n, pt)} for s, n, pc, pt in seg_defs]
control_n = sum(s["control_n"] for s in segments)
treatment_n = sum(s["treatment_n"] for s in segments)
experiments.append({
    "experiment_id": "new_pricing_page_layout", "name": "New Pricing Page Layout",
    "primary_metric": "trial_signup_rate", "hypothesis": "A benefits-led pricing page (vs. feature-table) increases trial signups.",
    "control_n": control_n, "control_conversions": sum(s["control_conversions"] for s in segments),
    "treatment_n": treatment_n, "treatment_conversions": sum(s["treatment_conversions"] for s in segments),
    "guardrails": [
        {"name": "day7_churn_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.095),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.093)},
        {"name": "support_tickets_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.11),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.112)},
    ],
    "segments": segments, "intended_split": 0.5,
})

# ============================================================
# 4. Search Ranking Algorithm v3 -- INVALID (Sample Ratio Mismatch)
# ============================================================
control_n, treatment_n = 5450, 4550  # a bucketing bug skewed the actual split away from the intended 50/50
control_conv = binom(control_n, 0.300)
treatment_conv = binom(treatment_n, 0.328)  # looks like a real, significant lift...
seg_defs = [("Starter", 0.55), ("Growth", 0.30), ("Enterprise", 0.15)]
segments = []
remaining_c, remaining_t = control_n, treatment_n
for i, (s, share) in enumerate(seg_defs):
    n_c = control_n if i == len(seg_defs) - 1 else int(control_n * share)
    n_t = treatment_n if i == len(seg_defs) - 1 else int(treatment_n * share)
    n_c, n_t = min(n_c, remaining_c), min(n_t, remaining_t)
    remaining_c -= n_c
    remaining_t -= n_t
    segments.append({"segment": s, "control_n": n_c, "control_conversions": binom(n_c, 0.300),
                      "treatment_n": n_t, "treatment_conversions": binom(n_t, 0.328)})
experiments.append({
    "experiment_id": "search_ranking_algo_v3", "name": "Search Ranking Algorithm v3",
    "primary_metric": "search_to_click_rate", "hypothesis": "A relevance-tuned ranking model increases search-result click-through.",
    "control_n": control_n, "control_conversions": control_conv,
    "treatment_n": treatment_n, "treatment_conversions": treatment_conv,
    "guardrails": [
        {"name": "day7_churn_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.088),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.086)},
        {"name": "support_tickets_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.10),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.099)},
    ],
    "segments": segments, "intended_split": 0.5,
})

# ============================================================
# 5. Dashboard Load Speed Optimization -- KILL (no effect)
# ============================================================
seg_defs = [("Starter", 2000, 0.55, 0.552), ("Growth", 1400, 0.56, 0.558), ("Enterprise", 600, 0.58, 0.575)]
segments = [{"segment": s, "control_n": n, "control_conversions": binom(n, pc),
             "treatment_n": n, "treatment_conversions": binom(n, pt)} for s, n, pc, pt in seg_defs]
control_n = sum(s["control_n"] for s in segments)
treatment_n = sum(s["treatment_n"] for s in segments)
experiments.append({
    "experiment_id": "dashboard_load_speed_opt", "name": "Dashboard Load Speed Optimization",
    "primary_metric": "week1_return_rate", "hypothesis": "A 40% dashboard load-time reduction increases week-1 return visits.",
    "control_n": control_n, "control_conversions": sum(s["control_conversions"] for s in segments),
    "treatment_n": treatment_n, "treatment_conversions": sum(s["treatment_conversions"] for s in segments),
    "guardrails": [
        {"name": "day7_churn_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.09),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.089)},
        {"name": "support_tickets_rate", "direction": "lower_is_better",
         "control_n": control_n, "control_events": binom(control_n, 0.10),
         "treatment_n": treatment_n, "treatment_events": binom(treatment_n, 0.101)},
    ],
    "segments": segments, "intended_split": 0.5,
})

with open("experiments.json", "w") as f:
    json.dump({"experiments": experiments}, f, indent=2)

for e in experiments:
    p_c = e["control_conversions"] / e["control_n"]
    p_t = e["treatment_conversions"] / e["treatment_n"]
    print(f"{e['experiment_id']:28s} n=({e['control_n']:,}/{e['treatment_n']:,})  "
          f"{e['primary_metric']}: {p_c:.3f} -> {p_t:.3f}  ({(p_t-p_c)*100:+.2f}pp)")

print("Done. Wrote experiments.json")
