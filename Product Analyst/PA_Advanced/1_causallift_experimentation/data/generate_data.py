"""
Generates three independent, realistic datasets for CausalLift, each
built to demonstrate a specific advanced experimentation/causal-inference
technique on Flowdesk (the same fictional B2B SaaS product used
elsewhere in this portfolio):

  1. exp_a_onboarding.csv — a randomized A/B test of an onboarding-flow
     redesign (n=12,000, true effects: +3.5pp activation, +continuous
     engagement lift), with a pre-experiment covariate for CUPED variance
     reduction.
  2. exp_b_daily_peeks.csv — 21 days of daily conversion counts for a
     pricing-page test with a TRUE NULL effect (no real difference),
     purpose-built to demonstrate the "peeking" false-positive-inflation
     problem and a Monte-Carlo-calibrated sequential correction.
  3. rollout_panel.csv — a region x week panel for a staged ("Smart
     Templates") feature rollout that could NOT be randomized (it shipped
     region-by-region), with a true +5pp causal lift in treated regions
     starting week 9, built for a difference-in-differences analysis.
"""
import numpy as np
import pandas as pd

# ============================================================
# 1. Experiment A -- onboarding redesign (randomized A/B test)
# ============================================================
rng = np.random.default_rng(101)
N_PER_ARM = 6000
n = N_PER_ARM * 2

pre_engagement_score = rng.normal(50, 15, n).clip(5, 100)
treatment = rng.integers(0, 2, size=n)  # properly randomized 50/50

# continuous secondary metric (day-7 engagement score) -- heavily
# explained by the pre-period covariate, which is exactly what makes
# CUPED valuable here
noise = rng.normal(0, 14, n)
day7_engagement = 12 + 0.85 * pre_engagement_score + treatment * 4.2 + noise
day7_engagement = day7_engagement.clip(0, None)

# primary binary metric (activation), correlated with day7 engagement
logit = -2.35 + 0.045 * day7_engagement
p_activate = 1 / (1 + np.exp(-logit))
activated = rng.binomial(1, p_activate)

exp_a = pd.DataFrame({
    "user_id": [f"U{100000+i}" for i in range(n)],
    "treatment": treatment,
    "pre_engagement_score": pre_engagement_score.round(2),
    "day7_engagement_score": day7_engagement.round(2),
    "activated": activated,
})
exp_a.to_csv("exp_a_onboarding.csv", index=False)

act_rates = exp_a.groupby("treatment").activated.mean()
eng_means = exp_a.groupby("treatment").day7_engagement_score.mean()
print(f"[Exp A] n={len(exp_a)} | activation: control={act_rates[0]:.3f} treatment={act_rates[1]:.3f} "
      f"(+{(act_rates[1]-act_rates[0])*100:.1f}pp) | day7 engagement: control={eng_means[0]:.1f} treatment={eng_means[1]:.1f}")

# ============================================================
# 2. Experiment B -- pricing page test, TRUE NULL, daily peeking demo
# ============================================================
K = 21
DAILY_N = 140
P_TRUE = 0.20  # identical in both arms -- no real effect

# search for a seed whose daily path is a genuine, undoctored instance of
# the peeking problem: naive daily-look testing flags "significant" at
# some day before day 21, purely by chance, under the true null.
chosen_seed = None
for seed in range(500):
    r = np.random.default_rng(seed)
    c_daily = r.binomial(DAILY_N, P_TRUE, K)
    t_daily = r.binomial(DAILY_N, P_TRUE, K)
    c_cum, t_cum = np.cumsum(c_daily), np.cumsum(t_daily)
    n_cum = np.arange(1, K + 1) * DAILY_N
    flagged_day = None
    for k in range(K):
        p_c, p_t = c_cum[k] / n_cum[k], t_cum[k] / n_cum[k]
        p_pool = (c_cum[k] + t_cum[k]) / (2 * n_cum[k])
        se = np.sqrt(2 * p_pool * (1 - p_pool) / n_cum[k])
        z = (p_t - p_c) / se if se > 0 else 0
        if abs(z) >= 1.96:
            flagged_day = k + 1
            break
    if flagged_day is not None and 6 <= flagged_day <= 16:
        chosen_seed = seed
        break

assert chosen_seed is not None, "no qualifying seed found -- widen search"
r = np.random.default_rng(chosen_seed)
c_daily = r.binomial(DAILY_N, P_TRUE, K)
t_daily = r.binomial(DAILY_N, P_TRUE, K)

exp_b = pd.DataFrame({
    "day": np.arange(1, K + 1),
    "control_signups": DAILY_N, "control_conversions": c_daily,
    "treatment_signups": DAILY_N, "treatment_conversions": t_daily,
})
exp_b.to_csv("exp_b_daily_peeks.csv", index=False)
print(f"[Exp B] seed={chosen_seed} (naive peeking flags a false positive by day {flagged_day}) | "
      f"K={K} days, {DAILY_N}/arm/day, true rate {P_TRUE} in both arms (true null)")

# ============================================================
# 3. Staged feature rollout panel -- diff-in-differences
# ============================================================
rng3 = np.random.default_rng(202)
REGIONS = ["Northeast", "Midwest", "South", "West"]
TREATED_REGIONS = {"South", "West"}
ROLLOUT_WEEK = 9
WEEKS = range(1, 17)
region_fe = {"Northeast": 0.0, "Midwest": -0.015, "South": 0.01, "West": -0.005}

rows = []
for region in REGIONS:
    base = 0.62 + region_fe[region]
    for w in WEEKS:
        trend = 0.0015 * w
        treat_effect = 0.0
        if region in TREATED_REGIONS and w >= ROLLOUT_WEEK:
            ramp = min(1.0, (w - ROLLOUT_WEEK + 1) / 2)
            treat_effect = 0.05 * ramp
        rate = base + trend + treat_effect + rng3.normal(0, 0.008)
        rows.append({
            "region": region, "week": w, "weekly_active_rate": round(rate, 4),
            "treated_region": region in TREATED_REGIONS, "post": w >= ROLLOUT_WEEK,
        })

panel = pd.DataFrame(rows)
panel.to_csv("rollout_panel.csv", index=False)
post_treated = panel[(panel.treated_region) & (panel.post)].weekly_active_rate.mean()
pre_treated = panel[(panel.treated_region) & (~panel.post)].weekly_active_rate.mean()
post_control = panel[(~panel.treated_region) & (panel.post)].weekly_active_rate.mean()
pre_control = panel[(~panel.treated_region) & (~panel.post)].weekly_active_rate.mean()
naive_did = (post_treated - pre_treated) - (post_control - pre_control)
print(f"[Rollout panel] {len(panel)} region-weeks | naive raw DiD (no fixed effects) ≈ {naive_did*100:.2f}pp "
      f"(true injected effect: 5.0pp, ramped over 2 weeks starting week {ROLLOUT_WEEK})")

print("Done. Wrote exp_a_onboarding.csv, exp_b_daily_peeks.csv, rollout_panel.csv")
