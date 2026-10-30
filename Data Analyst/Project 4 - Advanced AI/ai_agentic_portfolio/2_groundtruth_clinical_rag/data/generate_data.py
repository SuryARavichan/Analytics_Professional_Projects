"""
Generates 12,000 clinical-query events for Meridian Health AI, a
fictional AI clinical-documentation/care-coordination copilot that
answers clinician questions (medication dosing, drug interactions,
diagnosis criteria, care protocols) using RAG -- retrieval-augmented
generation over a medical knowledge base (drug formularies, clinical
guidelines, care pathways) -- rather than the model's own unverified
"knowledge." RAG is the dominant AI-native product architecture right
now precisely because it's supposed to reduce hallucination by grounding
answers in retrieved source text; this dataset is built to show that
grounding still fails in a specific, hideable way.

SIMPLIFICATION, stated honestly: a real RAG evaluation would score
retrieval against a labeled document corpus. This dataset instead
generates each query's retrieval outcome directly from a
per-segment/version "retrieval quality" probability (the chance each of
the top-5 retrieved chunks is actually relevant) plus a ground-truth
count of how many relevant chunks exist for that query -- enough to
compute real precision@k, recall@k, and MRR, without simulating an
actual document corpus and embedding space.

THE CORE STORY: Meridian upgrades its retrieval embedding model (v1 ->
v2). In AGGREGATE, retrieval quality improves for every query type,
including drug_interaction -- a clean, reportable win. But
drug_interaction queries about RARE/uncommon drug combinations (~15% of
that category's volume, underrepresented in whatever data the v2
embedding model was tuned on) actually get WORSE: retrieval quality
drops from 55% to 40%, driving a real spike in hallucination risk that
is completely hidden by the category's aggregate improvement (common
combinations dominate the volume-weighted average). Worse, physicians
are LESS likely to catch a hallucinated answer about a rare interaction
they're less personally familiar with -- so this is exactly the segment
where an ungrounded answer is both more likely to occur AND more likely
to silently slip through as "accepted as is." That combination -- a
segment-level regression hidden inside an aggregate win, colliding with
a segment-level drop in human-review catch rate -- is the patient-safety
finding this project is built to surface.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(909)

START = pd.Timestamp("2026-03-01")
N_DAYS = 50
DEPLOY_DAY = 25
DEPLOY_TS = START + pd.Timedelta(days=DEPLOY_DAY)
N_QUERIES = 24000
K_RETRIEVED = 5

QUERY_TYPES = ["medication_dosage", "drug_interaction", "diagnosis_criteria", "care_protocol"]
QUERY_TYPE_WEIGHT = {"medication_dosage": 0.35, "drug_interaction": 0.25, "diagnosis_criteria": 0.20, "care_protocol": 0.20}
RARE_COMBO_SHARE_OF_DRUG_INTERACTION = 0.20

N_RELEVANT_MEAN = {"medication_dosage": 4.0, "diagnosis_criteria": 3.0, "care_protocol": 5.0, "drug_interaction": 3.0}

# probability each of the K_RETRIEVED chunks is actually relevant, by segment and embedding version
RETRIEVAL_QUALITY = {
    ("medication_dosage", None, "v1"): 0.72, ("medication_dosage", None, "v2"): 0.82,
    ("diagnosis_criteria", None, "v1"): 0.68, ("diagnosis_criteria", None, "v2"): 0.78,
    ("care_protocol", None, "v1"): 0.70, ("care_protocol", None, "v2"): 0.75,
    ("drug_interaction", "common", "v1"): 0.65, ("drug_interaction", "common", "v2"): 0.80,
    ("drug_interaction", "rare", "v1"): 0.58, ("drug_interaction", "rare", "v2"): 0.32,  # the hidden regression
}

BASE_HALLUC_RATE = {"medication_dosage": 0.03, "drug_interaction": 0.03, "diagnosis_criteria": 0.025, "care_protocol": 0.02}
HALLUC_MULTIPLIER = 0.35  # additional hallucination probability at precision@k = 0

PHYSICIAN_CATCH_RATE = {"default": 0.55, "rare_combo": 0.35}  # rare interactions are less familiar, harder to catch


def business_day_weight(dow: int) -> float:
    return 0.5 if dow >= 5 else 1.0


day_weights = np.array([business_day_weight((START + pd.Timedelta(days=d)).dayofweek) for d in range(N_DAYS)])
day_offsets = rng.choice(np.arange(N_DAYS), size=N_QUERIES, p=day_weights / day_weights.sum())
timestamps = START + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(rng.integers(0, 24 * 3600, N_QUERIES), unit="s")

rows = []
for i in range(N_QUERIES):
    ts = timestamps[i]
    query_type = rng.choice(QUERY_TYPES, p=[QUERY_TYPE_WEIGHT[q] for q in QUERY_TYPES])
    version = "v2" if ts >= DEPLOY_TS else "v1"

    rare_combo = False
    if query_type == "drug_interaction":
        rare_combo = bool(rng.random() < RARE_COMBO_SHARE_OF_DRUG_INTERACTION)
        segment_key = (query_type, "rare" if rare_combo else "common", version)
    else:
        segment_key = (query_type, None, version)
    p_relevant = RETRIEVAL_QUALITY[segment_key]

    n_relevant_available = int(np.clip(round(rng.poisson(N_RELEVANT_MEAN[query_type])), 1, 8))
    num_relevant_retrieved = min(int(rng.binomial(K_RETRIEVED, p_relevant)), n_relevant_available)
    precision_at_k = num_relevant_retrieved / K_RETRIEVED
    recall_at_k = min(1.0, num_relevant_retrieved / n_relevant_available)
    if num_relevant_retrieved == 0:
        reciprocal_rank = 0.0
    else:
        first_relevant_rank = int(np.clip(rng.geometric(max(p_relevant, 0.05)), 1, K_RETRIEVED))
        reciprocal_rank = 1.0 / first_relevant_rank

    citation_coverage = float(np.clip(precision_at_k * 0.9 + rng.normal(0, 0.08), 0, 1))

    halluc_prob = np.clip(BASE_HALLUC_RATE[query_type] + (1 - precision_at_k) * HALLUC_MULTIPLIER, 0, 0.9)
    hallucination_occurred = bool(rng.random() < halluc_prob)

    physician_caught_issue = False
    if hallucination_occurred:
        catch_rate = PHYSICIAN_CATCH_RATE["rare_combo"] if rare_combo else PHYSICIAN_CATCH_RATE["default"]
        physician_caught_issue = bool(rng.random() < catch_rate)
        physician_action = rng.choice(["flagged_for_review", "rejected"], p=[0.6, 0.4]) if physician_caught_issue else "accepted_as_is"
    else:
        physician_action = rng.choice(["accepted_as_is", "edited"], p=[0.85, 0.15])

    rows.append({
        "query_id": f"Q{700000+i}", "timestamp": ts, "query_type": query_type,
        "rare_drug_combination": rare_combo, "embedding_version": version, "k_retrieved": K_RETRIEVED,
        "n_relevant_available": n_relevant_available, "num_relevant_retrieved": num_relevant_retrieved,
        "precision_at_k": round(precision_at_k, 3), "recall_at_k": round(recall_at_k, 3),
        "reciprocal_rank": round(reciprocal_rank, 3), "citation_coverage": round(citation_coverage, 3),
        "hallucination_occurred": hallucination_occurred, "physician_caught_issue": physician_caught_issue,
        "physician_action": physician_action,
    })

queries = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
queries.to_csv("queries.csv", index=False)

print(f"{len(queries):,} queries | {queries.timestamp.min()} to {queries.timestamp.max()} | deploy at day {DEPLOY_DAY} ({DEPLOY_TS.date()})")
for version in ["v1", "v2"]:
    sub = queries[queries.embedding_version == version]
    print(f"\n--- {version} ---")
    g = sub.groupby(["query_type", "rare_drug_combination"]).agg(
        n=("query_id", "count"), precision_at_k=("precision_at_k", "mean"),
        recall_at_k=("recall_at_k", "mean"), mrr=("reciprocal_rank", "mean"),
        hallucination_rate=("hallucination_occurred", "mean"),
        silent_failure_rate=("physician_action", lambda s: ((s == "accepted_as_is") & sub.loc[s.index, "hallucination_occurred"]).mean()),
    ).round(4)
    print(g)
print("\nWrote queries.csv")
