"""
anomaly_detector.py — two complementary statistical detectors, because one
technique doesn't catch every anomaly shape:

  1. point_anomaly_detector(): STL seasonal decomposition (7-day period, to
     absorb day-of-week seasonality -- Flowdesk usage is heavily weekday-
     skewed, a B2B pattern) + z-score on the residual. Good at catching
     sharp single/multi-day spikes and drops in total daily adoption
     events -- misses slow drift because a gradual decline never produces
     a single extreme residual day.
  2. channel_drift_detector(): compares each channel's trailing 7-day
     average activation rate (activations/signups) to its 7-day average
     from 3 weeks prior; flags a sustained decline, not a single bad day.
     This is the detector that catches a slow-building channel-quality
     problem the point-anomaly detector would miss entirely -- an
     activation-rate drift doesn't move total adoption-event *volume*
     enough to trip a volume-based z-score.

Using the right detector for the anomaly shape -- rather than cranking one
threshold until it catches everything -- is itself the point being
demonstrated here.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def point_anomaly_detector(daily_totals: pd.Series, z_threshold: float = 3.5) -> pd.DataFrame:
    """daily_totals: a Series of daily adoption-event counts indexed by date (DatetimeIndex)."""
    stl = STL(daily_totals, period=7, robust=True)
    result = stl.fit()
    resid = result.resid
    z = (resid - resid.mean()) / resid.std()

    out = pd.DataFrame({
        "date": daily_totals.index, "actual": daily_totals.values,
        "trend": result.trend.values, "seasonal": result.seasonal.values,
        "residual": resid.values, "z_score": z.values,
    })
    out["is_anomaly"] = out.z_score.abs() >= z_threshold
    out["direction"] = np.where(out.z_score > 0, "spike", "drop")
    return out


def channel_drift_detector(funnel_df: pd.DataFrame, decline_threshold: float = -0.20,
                            min_consecutive_days: int = 5,
                            exclude_windows: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """funnel_df: long-format daily_channel_funnel with columns
    [date, channel, signups, activations]. Flags the first date in any run
    of >= min_consecutive_days where a channel's trailing-7-day average
    activation rate has declined more than `decline_threshold` versus its
    trailing-7-day average from 21 days earlier.

    `exclude_windows`: date ranges to exclude from flagging (e.g. a known
    holiday-adjacent signup lull, where a fixed 21-day-prior comparison
    isn't seasonally robust). That's a real, documented limitation of this
    detector, not hidden here -- a production version would compare
    against a seasonally-adjusted baseline instead of a raw N-days-ago
    snapshot."""
    df = funnel_df.copy()
    df["activation_rate"] = df.activations / df.signups
    pivot = df.pivot(index="date", columns="channel", values="activation_rate").sort_index()
    roll7 = pivot.rolling(7).mean()
    roll7_lag21 = roll7.shift(21)
    pct_change = (roll7 - roll7_lag21) / roll7_lag21

    exclude_windows = exclude_windows or []
    excluded_mask = pd.Series(False, index=pivot.index)
    for start, end in exclude_windows:
        excluded_mask |= (pivot.index >= pd.Timestamp(start)) & (pivot.index <= pd.Timestamp(end))

    flags = []
    for channel in pivot.columns:
        declining = (pct_change[channel] <= decline_threshold) & (~excluded_mask)
        run_id = (declining != declining.shift()).cumsum()
        for _, run in declining.groupby(run_id):
            if run.iloc[0] and len(run) >= min_consecutive_days:
                start_date = run.index[0]
                trough_pct = pct_change[channel].loc[run.index].min()
                flags.append({
                    "channel": channel, "drift_start_date": start_date,
                    "run_length_days": len(run), "max_decline_pct": round(trough_pct * 100, 1),
                })
    return pd.DataFrame(flags)
