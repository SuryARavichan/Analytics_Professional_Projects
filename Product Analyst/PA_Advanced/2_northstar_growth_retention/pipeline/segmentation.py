"""
segmentation.py — unsupervised engagement segmentation (k-means on
standardized recency/frequency/breadth features) for currently-active
accounts, labeled dynamically from each cluster's actual centroid values
rather than a fixed rule-based tier system -- so the segment labels
reflect what the data actually contains, the same "label the cluster from
its real stats" pattern used for theme discovery elsewhere in this
portfolio, applied here to numeric behavioral data instead of text.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


RFM_COLS = ["recency_weeks_since_active", "recent_4wk_actions", "features_used_recent"]


def segment_active_accounts(accounts: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    active = accounts[accounts.event_observed == 0].copy()  # not yet churned as of the anchor date
    X = active[RFM_COLS].values
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    active["cluster"] = km.fit_predict(X_scaled)

    centroids = active.groupby("cluster")[RFM_COLS].mean()
    # rank clusters by a simple "engagement score" (low recency + high frequency + high breadth)
    centroids["engagement_score"] = (
        -centroids.recency_weeks_since_active.rank() + centroids.recent_4wk_actions.rank() + centroids.features_used_recent.rank()
    )
    ranked = centroids.sort_values("engagement_score", ascending=False)
    labels_in_order = ["Power User", "Regular", "Casual", "At-Risk (Active but Fading)"]
    label_map = {cluster_id: labels_in_order[i] for i, cluster_id in enumerate(ranked.index[:len(labels_in_order)])}
    active["segment"] = active.cluster.map(label_map)
    return active, centroids.loc[list(label_map.keys())].assign(segment=[label_map[c] for c in label_map])
