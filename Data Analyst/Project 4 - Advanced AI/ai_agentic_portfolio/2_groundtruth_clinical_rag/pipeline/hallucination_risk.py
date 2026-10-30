"""
hallucination_risk.py — the patient-safety-specific metric this project
adds on top of retrieval quality: SILENT FAILURE RATE, the share of
queries where a hallucination occurred AND the physician accepted the
answer as-is (didn't catch it). This is the metric that actually matters
for patient safety -- a hallucination that gets caught and rejected is a
near-miss; a hallucination that gets silently accepted is a real error
that reaches a clinical decision.
"""
import pandas as pd


def hallucination_summary(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    return df.groupby(groupby).agg(
        n=("query_id", "count"),
        hallucination_rate=("hallucination_occurred", "mean"),
        physician_catch_rate=("physician_caught_issue", lambda s: s[df.loc[s.index, "hallucination_occurred"]].mean()
                               if df.loc[s.index, "hallucination_occurred"].any() else float("nan")),
    ).round(4)


def silent_failure_rate(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["silent_failure"] = d.hallucination_occurred & (d.physician_action == "accepted_as_is")
    return d.groupby(groupby).agg(
        n=("query_id", "count"), silent_failure_rate=("silent_failure", "mean"),
    ).round(4)
