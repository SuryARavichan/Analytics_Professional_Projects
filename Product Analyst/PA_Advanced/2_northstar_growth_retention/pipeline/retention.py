"""
retention.py — thin helpers around the SQL cohort-retention query:
pivots the long SQL output into a cohort x milestone-week matrix for the
heatmap, and computes a blended (all-cohort) retention curve restricted
the same way (only cohorts mature enough to have reached each milestone
week are included in that week's denominator).
"""
import sqlite3
import pandas as pd


def load_cohort_retention(db_path: str, sql_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    return pd.read_sql_query(open(sql_path).read(), conn)


def cohort_pivot(cohort_df: pd.DataFrame) -> pd.DataFrame:
    return cohort_df.pivot(index="signup_month", columns="milestone_week", values="retention_pct").sort_index()


def blended_retention_curve(cohort_df: pd.DataFrame) -> pd.DataFrame:
    agg = cohort_df.groupby("milestone_week").agg(
        eligible_accounts=("eligible_accounts", "sum"), retained_accounts=("retained_accounts", "sum")
    ).reset_index()
    agg["retention_pct"] = round(100 * agg.retained_accounts / agg.eligible_accounts, 1)
    return agg
