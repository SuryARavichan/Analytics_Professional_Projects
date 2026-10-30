"""
themes.py — unsupervised theme discovery via TF-IDF + KMeans. Deliberately
NOT a keyword-tagging system (that requires knowing the themes in advance,
which defeats the point of "discover what customers are actually talking
about") -- this clusters reviews purely from their text and only afterward
extracts the top distinguishing terms per cluster, which is what an LLM
(or a human) then reads to assign a human-readable label.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

CUSTOM_STOPWORDS = [
    "app", "just", "really", "ve", "don", "didn", "doesn", "isn", "wasn",
    "time", "use", "using", "used", "got", "get", "gets",
    "honestly", "overall", "far", "months", "keeps", "single", "way",
    "basically", "far syncs", "overall syncs", "honestly syncs",
]


def discover_themes(texts: pd.Series, n_clusters: int = 7, random_state: int = 42):
    vectorizer = TfidfVectorizer(
        max_features=3000, stop_words="english", ngram_range=(1, 1),
        min_df=8, max_df=0.4,
    )
    X = vectorizer.fit_transform(texts)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    top_terms_per_cluster = {}
    for c in range(n_clusters):
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:12]
        top_terms_per_cluster[c] = [terms[i] for i in top_idx
                                     if terms[i] not in CUSTOM_STOPWORDS][:10]

    return cluster_labels, top_terms_per_cluster, vectorizer, km
