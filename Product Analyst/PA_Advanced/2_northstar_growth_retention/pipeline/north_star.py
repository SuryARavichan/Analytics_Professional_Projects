"""
north_star.py — a "magic number" style analysis (the Facebook "7 friends
in 10 days" / Slack "2,000 messages" tradition): given several candidate
early-usage metrics, which one is actually the best LEADING INDICATOR of
long-term retention? Picking a North Star metric without checking this is
picking a metric because it sounds plausible, not because it predicts
anything.

Method: restrict to accounts mature enough to have an observed 12-week
retention outcome (avoiding survivorship/immaturity bias from including
accounts too young to have reached week 12 yet), then for each candidate
metric fit a univariate logistic regression predicting week-12 retention
and report both the correlation and the ROC-AUC -- correlation alone can
be misleading for a binary outcome, AUC is the standard metric for "how
good is this signal at ranking accounts by outcome."
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def mature_cohort_with_outcome(accounts: pd.DataFrame, retention_week: int = 12) -> pd.DataFrame:
    mature = accounts[accounts.observed_horizon >= retention_week].copy()
    mature["retained_outcome"] = (mature.duration_weeks >= retention_week).astype(int)
    return mature


def evaluate_candidates(mature: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = mature["retained_outcome"].values
    for col in candidate_cols:
        x = mature[[col]].values
        corr = np.corrcoef(mature[col], y)[0, 1]
        clf = LogisticRegression()
        clf.fit(x, y)
        auc = roc_auc_score(y, clf.predict_proba(x)[:, 1])
        rows.append({"candidate_metric": col, "correlation_with_wk12_retention": round(corr, 3),
                      "auc": round(auc, 3)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
