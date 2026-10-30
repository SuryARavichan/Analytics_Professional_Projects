"""
retrieval_metrics.py — standard information-retrieval evaluation metrics
(precision@k, recall@k, MRR) computed on GroundTruth Clinical's query
log, the specific measurement that distinguishes a RAG-quality analysis
from a generic "quality score" analysis: it evaluates the RETRIEVAL step
directly, not just the final answer, which is what lets you localize a
quality problem to "the wrong documents were retrieved" versus "the
right documents were retrieved but the model ignored them."
"""
import numpy as np
import pandas as pd
from scipy import stats


def retrieval_summary(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    return df.groupby(groupby).agg(
        n=("query_id", "count"),
        precision_at_k=("precision_at_k", "mean"),
        recall_at_k=("recall_at_k", "mean"),
        mrr=("reciprocal_rank", "mean"),
        hallucination_rate=("hallucination_occurred", "mean"),
    ).round(4)


def compare_versions_ttest(df: pd.DataFrame, metric: str, filter_query_type: str | None = None,
                            filter_rare: bool | None = None) -> dict:
    """Independent two-sample t-test comparing a continuous retrieval
    metric (precision_at_k, recall_at_k, or reciprocal_rank) between
    embedding_version v1 and v2, optionally filtered to one query_type
    and/or one rare_drug_combination flag -- this is what lets the same
    function test both the (misleadingly stable) drug_interaction
    aggregate AND the rare-combination sub-segment specifically."""
    sub = df
    if filter_query_type is not None:
        sub = sub[sub.query_type == filter_query_type]
    if filter_rare is not None:
        sub = sub[sub.rare_drug_combination == filter_rare]

    v1 = sub[sub.embedding_version == "v1"][metric]
    v2 = sub[sub.embedding_version == "v2"][metric]
    t, p = stats.ttest_ind(v1, v2, equal_var=False)
    return {
        "metric": metric, "query_type": filter_query_type, "rare_drug_combination": filter_rare,
        "n_v1": len(v1), "n_v2": len(v2), "mean_v1": round(v1.mean(), 4), "mean_v2": round(v2.mean(), 4),
        "diff": round(v2.mean() - v1.mean(), 4), "t_stat": round(t, 3), "p_value": p,
        "significant": bool(p < 0.01),
    }


def two_proportion_ztest(p1_successes: int, p1_n: int, p2_successes: int, p2_n: int, alpha: float = 0.05) -> dict:
    p1, p2 = p1_successes / p1_n, p2_successes / p2_n
    p_pool = (p1_successes + p2_successes) / (p1_n + p2_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / p1_n + 1 / p2_n))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "p1": round(p1, 4), "p2": round(p2, 4), "diff_pp": round((p2 - p1) * 100, 2),
        "z": round(z, 3), "p_value": p_value, "significant": bool(p_value < alpha),
    }
