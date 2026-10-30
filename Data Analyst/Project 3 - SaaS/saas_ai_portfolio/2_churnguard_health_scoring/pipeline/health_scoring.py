"""
health_scoring.py — the two-sided health-scoring core of ChurnGuard AI.

Most "churn prediction" portfolio projects stop at one model: will this
account leave? That's half the picture for a real SaaS revenue team.
This module trains TWO models on the same leakage-free feature set
(sql/02_feature_engineering.sql):

  1. churn_risk_score    — P(account churns), a down-risk signal
  2. expansion_score      — P(account is an expansion/upsell candidate),
                            an up-opportunity signal

...and combines them into a 2x2 HEALTH QUADRANT, because "at risk" and
"expansion candidate" call for completely different actions from
completely different teams (Customer Success vs. Sales), and a single
blended "health score" would hide that.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["nps_missing"] = df["nps_score"].isna().astype(int)
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    df["util_trend_pct_3mo"] = df["util_trend_pct_3mo"].fillna(0)

    cat_cols = ["plan_tier", "contract_type", "region"]
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    drop_cols = ["account_id", "industry", "churned", "expanded"]
    feature_cols = [c for c in df_model.columns if c not in drop_cols]
    return df_model, feature_cols


def train_model(df_model: pd.DataFrame, feature_cols: list, target_col: str, random_state: int = 42):
    X = df_model[feature_cols]
    y = df_model[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=12,
                                  class_weight="balanced", random_state=random_state)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return clf, auc


def assign_quadrant(churn_risk: float, expansion_score: float,
                     risk_threshold: float = 0.35, expansion_threshold: float = 0.35) -> str:
    high_risk = churn_risk >= risk_threshold
    high_expansion = expansion_score >= expansion_threshold
    if high_risk and not high_expansion:
        return "At Risk — Save"
    if high_risk and high_expansion:
        return "Engaged but Unstable"     # using it heavily, but still showing risk signals -- worth a closer look
    if not high_risk and high_expansion:
        return "Growth Opportunity"
    return "Stable — Maintain"
