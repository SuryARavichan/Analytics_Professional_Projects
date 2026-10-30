"""
economics.py — cost-per-ticket analysis, so a containment-metric finding
can be paired with its actual cost impact rather than assumed. The key
question for account_recovery specifically: did the v2 policy's official
containment "win" actually save money once the downstream cost of
reopened tickets (many of which need a human anyway) is counted?
"""
import pandas as pd


def cost_by_path(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("resolution_path").agg(
        n=("ticket_id", "count"), avg_cost_usd=("cost_usd", "mean"), total_cost_usd=("cost_usd", "sum"),
    ).round(2)


def cost_by_category_version(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["category", "agent_version"]).agg(
        n=("ticket_id", "count"), avg_cost_usd=("cost_usd", "mean"), total_cost_usd=("cost_usd", "sum"),
    ).round(2)


def reopen_cost_impact(df: pd.DataFrame, category: str) -> dict:
    """Isolates the specific cost contribution of reopened tickets within
    one category/version, so the notebook can show whether a containment
    policy change actually reduced total cost or just moved when the
    cost gets paid (upfront escalation vs. a later reopen-triggered one)."""
    out = {}
    for version in ["v1", "v2"]:
        sub = df[(df.category == category) & (df.agent_version == version)]
        contained = sub[sub.resolution_path == "agent_resolved"]
        reopened = contained[contained.reopened_within_48h]
        out[version] = {
            "n_tickets": len(sub),
            "avg_cost_all": round(sub.cost_usd.mean(), 2),
            "n_contained": len(contained),
            "n_reopened": len(reopened),
            "reopen_rate": round(len(reopened) / len(contained), 4) if len(contained) else float("nan"),
            "avg_cost_reopened_tickets": round(reopened.cost_usd.mean(), 2) if len(reopened) else float("nan"),
            "avg_cost_non_reopened_contained": round(contained[~contained.reopened_within_48h].cost_usd.mean(), 2) if len(contained) else float("nan"),
        }
    return out
