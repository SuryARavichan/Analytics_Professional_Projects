"""
survival.py — Kaplan-Meier survival analysis (via `lifelines`) for
time-to-churn, comparing onboarding paths. This is a genuinely different
statistical tool from the classification-based churn scoring used
elsewhere in this portfolio (see the companion ChurnGuard AI project): a
survival model uses the CENSORED accounts (still active as of the anchor
date, so their true time-to-churn is unknown but at least this long)
correctly, rather than discarding them or mislabeling them as
"not-churned", and produces a full time-to-event curve rather than a
single point-in-time probability.
"""
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def fit_km_by_group(accounts: pd.DataFrame, group_col: str, duration_col: str = "duration_weeks",
                     event_col: str = "event_observed") -> dict:
    fitters = {}
    for group_value, sub in accounts.groupby(group_col):
        kmf = KaplanMeierFitter(label=str(group_value))
        kmf.fit(durations=sub[duration_col], event_observed=sub[event_col])
        fitters[group_value] = kmf
    return fitters


def compare_groups_logrank(accounts: pd.DataFrame, group_col: str, group_a, group_b,
                            duration_col: str = "duration_weeks", event_col: str = "event_observed") -> dict:
    a = accounts[accounts[group_col] == group_a]
    b = accounts[accounts[group_col] == group_b]
    result = logrank_test(a[duration_col], b[duration_col], event_observed_A=a[event_col], event_observed_B=b[event_col])
    return {"group_a": group_a, "group_b": group_b, "test_statistic": result.test_statistic, "p_value": result.p_value}


def median_survival_by_group(fitters: dict) -> pd.DataFrame:
    rows = []
    for group_value, kmf in fitters.items():
        rows.append({"group": group_value, "median_survival_weeks": kmf.median_survival_time_,
                     "n": int(kmf.event_table.at_risk.iloc[0])})
    return pd.DataFrame(rows)
