"""
sequential_testing.py — quantifies and fixes the "peeking problem": when a
team checks a fixed-horizon experiment's p-value every day and stops as
soon as it crosses 0.05, the TRUE false-positive rate is much higher than
5%, because each daily look is another chance for noise to cross the
threshold (this is the same statistical issue behind Simmons et al.'s
"optional stopping" and the industry-standard fix behind "always valid
p-values" / sequential testing platforms).

This module:
  1. simulate_naive_peeking() — Monte Carlo estimate of the TRUE type-I
     error rate when naively checking a fixed 1.96 z-threshold at every
     one of K daily looks, under the null.
  2. calibrate_sequential_boundary() — Monte-Carlo-calibrates a single
     constant `c` for an O'Brien-Fleming-shaped boundary schedule
     z_boundary(k) = c * sqrt(K/k) (conservative early, approaches the
     nominal threshold near the end), searching for the value of c that
     controls the TRUE type-I error rate at the target alpha across the
     same K daily looks -- this is a legitimate, standard way to calibrate
     a group-sequential boundary when you want an exact, simulation-
     verified error rate rather than relying on a textbook table.
  3. monitor_daily() — applies a boundary schedule to one real experiment's
     day-by-day data and reports the first (if any) day it's crossed.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats as _stats


def daily_zscores(control_daily: np.ndarray, treatment_daily: np.ndarray, n_per_day: int) -> np.ndarray:
    c_cum = np.cumsum(control_daily)
    t_cum = np.cumsum(treatment_daily)
    n_cum = np.arange(1, len(control_daily) + 1) * n_per_day
    p_c = c_cum / n_cum
    p_t = t_cum / n_cum
    p_pool = (c_cum + t_cum) / (2 * n_cum)
    se = np.sqrt(2 * p_pool * (1 - p_pool) / n_cum)
    z = np.where(se > 0, (p_t - p_c) / se, 0.0)
    return z


def simulate_naive_peeking(K: int, n_per_day: int, p_true: float, n_sims: int = 5000,
                            z_threshold: float = 1.96, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    any_flag = 0
    flag_days = []
    for _ in range(n_sims):
        c = rng.binomial(n_per_day, p_true, K)
        t = rng.binomial(n_per_day, p_true, K)
        z = daily_zscores(c, t, n_per_day)
        crossed = np.where(np.abs(z) >= z_threshold)[0]
        if len(crossed) > 0:
            any_flag += 1
            flag_days.append(crossed[0] + 1)
    return {
        "n_sims": n_sims, "K": K, "true_false_positive_rate": any_flag / n_sims,
        "nominal_alpha": 2 * _stats.norm.sf(z_threshold),
        "median_first_flag_day": float(np.median(flag_days)) if flag_days else None,
    }


def calibrate_sequential_boundary(K: int, n_per_day: int, p_true: float, target_alpha: float = 0.05,
                                   n_sims: int = 5000, seed: int = 1, c_lo: float = 1.5, c_hi: float = 4.5,
                                   tol: float = 0.003, max_iter: int = 25) -> dict:
    """Bisection search for the constant c in z_boundary(k) = c*sqrt(K/k)
    that controls the true type-I error rate at target_alpha, evaluated
    via Monte Carlo at each candidate c."""

    def _fpr_for_c(c: float) -> float:
        rng = np.random.default_rng(seed)
        boundary = c * np.sqrt(K / np.arange(1, K + 1))
        any_flag = 0
        for _ in range(n_sims):
            ctrl = rng.binomial(n_per_day, p_true, K)
            trt = rng.binomial(n_per_day, p_true, K)
            z = daily_zscores(ctrl, trt, n_per_day)
            if np.any(np.abs(z) >= boundary):
                any_flag += 1
        return any_flag / n_sims

    lo, hi = c_lo, c_hi
    fpr_hi = _fpr_for_c(hi)
    fpr_lo = _fpr_for_c(lo)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fpr_mid = _fpr_for_c(mid)
        if abs(fpr_mid - target_alpha) <= tol:
            lo = hi = mid
            break
        # higher c -> stricter boundary -> lower false-positive rate
        if fpr_mid > target_alpha:
            lo = mid
        else:
            hi = mid
    c_final = (lo + hi) / 2
    achieved_fpr = _fpr_for_c(c_final)
    return {"c": c_final, "K": K, "achieved_false_positive_rate": achieved_fpr,
            "target_alpha": target_alpha, "boundary_schedule": (c_final * np.sqrt(K / np.arange(1, K + 1))).tolist()}


@dataclass
class MonitorResult:
    z_scores: np.ndarray
    naive_flag_day: int | None
    calibrated_flag_day: int | None
    calibrated_boundary: np.ndarray


def monitor_daily(control_daily: np.ndarray, treatment_daily: np.ndarray, n_per_day: int,
                   calibrated_c: float, naive_threshold: float = 1.96) -> MonitorResult:
    K = len(control_daily)
    z = daily_zscores(control_daily, treatment_daily, n_per_day)
    boundary = calibrated_c * np.sqrt(K / np.arange(1, K + 1))

    naive_crossed = np.where(np.abs(z) >= naive_threshold)[0]
    calib_crossed = np.where(np.abs(z) >= boundary)[0]

    return MonitorResult(
        z_scores=z,
        naive_flag_day=int(naive_crossed[0] + 1) if len(naive_crossed) else None,
        calibrated_flag_day=int(calib_crossed[0] + 1) if len(calib_crossed) else None,
        calibrated_boundary=boundary,
    )
