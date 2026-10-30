"""
diagnostics.py — the "tools" SentinelBI's agent calls to investigate a
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


def breakdown_by_region(db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
        SELECT region, ROUND(SUM(revenue),2) AS revenue
        FROM daily_metrics WHERE date BETWEEN ? AND ?
        GROUP BY region ORDER BY revenue DESC;
    """
    return pd.read_sql_query(q, _conn(db_path), params=(start_date, end_date))


def breakdown_by_channel(db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
        SELECT channel, ROUND(SUM(revenue),2) AS revenue
        FROM daily_metrics WHERE date BETWEEN ? AND ?
        GROUP BY channel ORDER BY revenue DESC;
    """
    return pd.read_sql_query(q, _conn(db_path), params=(start_date, end_date))


def compare_to_baseline(db_path: str, start_date: str, end_date: str,
                         baseline_start: str, baseline_end: str,
                         group_col: str | None = None) -> pd.DataFrame:
    """Compares total revenue in [start_date, end_date] against a baseline
    window, optionally broken out by `group_col` (e.g. 'region' or
    'channel'), so the agent can see WHICH slice moved, not just that the
    total moved."""
    if group_col:
        q = f"""
            SELECT {group_col},
                   ROUND(SUM(CASE WHEN date BETWEEN ? AND ? THEN revenue ELSE 0 END),2) AS incident_revenue,
                   ROUND(SUM(CASE WHEN date BETWEEN ? AND ? THEN revenue ELSE 0 END),2) AS baseline_revenue
            FROM daily_metrics
            WHERE date BETWEEN ? AND ? OR date BETWEEN ? AND ?
            GROUP BY {group_col};
        """
        params = (start_date, end_date, baseline_start, baseline_end,
                  start_date, end_date, baseline_start, baseline_end)
    else:
        q = """
            SELECT
                ROUND(SUM(CASE WHEN date BETWEEN ? AND ? THEN revenue ELSE 0 END),2) AS incident_revenue,
                ROUND(SUM(CASE WHEN date BETWEEN ? AND ? THEN revenue ELSE 0 END),2) AS baseline_revenue
            FROM daily_metrics
            WHERE date BETWEEN ? AND ? OR date BETWEEN ? AND ?;
        """
        params = (start_date, end_date, baseline_start, baseline_end,
                  start_date, end_date, baseline_start, baseline_end)
    df = pd.read_sql_query(q, _conn(db_path), params=params)
    n_incident_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    n_baseline_days = (pd.Timestamp(baseline_end) - pd.Timestamp(baseline_start)).days + 1
    df["incident_daily_avg"] = round(df["incident_revenue"] / n_incident_days, 2)
    df["baseline_daily_avg"] = round(df["baseline_revenue"] / n_baseline_days, 2)
    df["pct_change"] = round(100 * (df["incident_daily_avg"] - df["baseline_daily_avg"]) / df["baseline_daily_avg"], 1)
    return df
