"""
diagnostics.py — the "tools" LaunchPulse's agent calls to investigate a
flagged anomaly. Each function is a real, parameterized SQL query against
the SQLite database (not a mock) -- this is the deterministic diagnostic
playbook that runs automatically the moment an anomaly is detected, before
the LLM ever sees the incident. The LLM's job (see agent.py) is to
synthesize what these queries already found, not to guess at a cause.
"""
import sqlite3
import pandas as pd


def _conn(db_path):
    return sqlite3.connect(db_path)


def breakdown_by_feature(db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
        SELECT feature_name, SUM(adoption_events) AS adoption_events
        FROM daily_metrics WHERE date BETWEEN ? AND ?
        GROUP BY feature_name ORDER BY adoption_events DESC;
    """
    return pd.read_sql_query(q, _conn(db_path), params=(start_date, end_date))


def breakdown_by_channel(db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
        SELECT channel, SUM(adoption_events) AS adoption_events
        FROM daily_metrics WHERE date BETWEEN ? AND ?
        GROUP BY channel ORDER BY adoption_events DESC;
    """
    return pd.read_sql_query(q, _conn(db_path), params=(start_date, end_date))


def compare_to_baseline(db_path: str, start_date: str, end_date: str,
                         baseline_start: str, baseline_end: str,
                         group_col: str | None = None) -> pd.DataFrame:
    """Compares total adoption_events in [start_date, end_date] against a
    baseline window, optionally broken out by `group_col` (e.g.
    'feature_name' or 'channel'), so the agent can see WHICH slice moved,
    not just that the total moved."""
    if group_col:
        q = f"""
            SELECT {group_col},
                   SUM(CASE WHEN date BETWEEN ? AND ? THEN adoption_events ELSE 0 END) AS incident_events,
                   SUM(CASE WHEN date BETWEEN ? AND ? THEN adoption_events ELSE 0 END) AS baseline_events
            FROM daily_metrics
            WHERE date BETWEEN ? AND ? OR date BETWEEN ? AND ?
            GROUP BY {group_col};
        """
        params = (start_date, end_date, baseline_start, baseline_end,
                  start_date, end_date, baseline_start, baseline_end)
    else:
        q = """
            SELECT
                SUM(CASE WHEN date BETWEEN ? AND ? THEN adoption_events ELSE 0 END) AS incident_events,
                SUM(CASE WHEN date BETWEEN ? AND ? THEN adoption_events ELSE 0 END) AS baseline_events
            FROM daily_metrics
            WHERE date BETWEEN ? AND ? OR date BETWEEN ? AND ?;
        """
        params = (start_date, end_date, baseline_start, baseline_end,
                  start_date, end_date, baseline_start, baseline_end)
    df = pd.read_sql_query(q, _conn(db_path), params=params)
    n_incident_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    n_baseline_days = (pd.Timestamp(baseline_end) - pd.Timestamp(baseline_start)).days + 1
    df["incident_daily_avg"] = round(df["incident_events"] / n_incident_days, 1)
    df["baseline_daily_avg"] = round(df["baseline_events"] / n_baseline_days, 1)
    df["pct_change"] = round(100 * (df["incident_daily_avg"] - df["baseline_daily_avg"]) / df["baseline_daily_avg"], 1)
    return df


def channel_funnel_breakdown(db_path: str, start_date: str, end_date: str,
                              baseline_start: str, baseline_end: str) -> pd.DataFrame:
    """Per-channel signups, activations, and activation rate in the
    incident window vs. a baseline window -- the specific diagnostic for a
    channel_drift incident, since a funnel problem shows up here, not in
    feature-usage volume."""
    q = """
        SELECT channel,
               SUM(CASE WHEN date BETWEEN ? AND ? THEN signups ELSE 0 END)     AS incident_signups,
               SUM(CASE WHEN date BETWEEN ? AND ? THEN activations ELSE 0 END) AS incident_activations,
               SUM(CASE WHEN date BETWEEN ? AND ? THEN signups ELSE 0 END)     AS baseline_signups,
               SUM(CASE WHEN date BETWEEN ? AND ? THEN activations ELSE 0 END) AS baseline_activations
        FROM daily_channel_funnel
        WHERE date BETWEEN ? AND ? OR date BETWEEN ? AND ?
        GROUP BY channel;
    """
    params = (start_date, end_date, start_date, end_date, baseline_start, baseline_end,
              baseline_start, baseline_end, start_date, end_date, baseline_start, baseline_end)
    df = pd.read_sql_query(q, _conn(db_path), params=params)
    df["incident_activation_rate"] = round(100 * df.incident_activations / df.incident_signups, 1)
    df["baseline_activation_rate"] = round(100 * df.baseline_activations / df.baseline_signups, 1)
    df["activation_rate_pp_change"] = round(df.incident_activation_rate - df.baseline_activation_rate, 1)
    return df
