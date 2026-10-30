"""
call_anomaly.py — the POINT-ANOMALY detector for a shared upstream ASR
(speech-recognition) provider incident: STL seasonal decomposition +
z-score on the residual, applied to hourly mean transcription confidence
across ALL regions/call types combined -- a shared-infrastructure
incident should hit everyone at once, which is exactly what
distinguishes it from the region-specific dialect regression this
project's headline finding is about.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def hourly_confidence_series(calls: pd.DataFrame) -> pd.Series:
    return calls.set_index("timestamp").resample("h").transcription_confidence.mean().interpolate()


def point_anomaly_detector(hourly_series: pd.Series, z_threshold: float = 3.5, period: int = 24) -> pd.DataFrame:
    stl = STL(hourly_series, period=period, robust=True)
    result = stl.fit()
    resid = result.resid
    z = (resid - resid.mean()) / resid.std()
    out = pd.DataFrame({
        "timestamp": hourly_series.index, "actual": hourly_series.values,
        "residual": resid.values, "z_score": z.values,
    })
    out["is_anomaly"] = out.z_score.abs() >= z_threshold
    out["direction"] = np.where(out.z_score > 0, "spike", "drop")
    return out


def find_anomaly_windows(anomaly_df: pd.DataFrame, min_consecutive_hours: int = 2) -> pd.DataFrame:
    flagged = anomaly_df.is_anomaly
    run_id = (flagged != flagged.shift()).cumsum()
    windows = []
    for _, run in anomaly_df.groupby(run_id):
        if run.is_anomaly.iloc[0] and len(run) >= min_consecutive_hours:
            windows.append({
                "start": run.timestamp.iloc[0], "end": run.timestamp.iloc[-1],
                "duration_hours": len(run), "peak_z_score": run.z_score.abs().max(),
                "trough_confidence": run.actual.min(), "direction": run.direction.iloc[0],
            })
    return pd.DataFrame(windows)
