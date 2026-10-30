"""
Generates 20,000 voice-AI calls for Portside AI, a fictional logistics
AI platform whose voice agent calls truck drivers for ETA check-ins,
exception reports, proof-of-delivery confirmation, and reschedule
requests -- the "voice/multimodal AI agent" trend: an agent that
conducts a live spoken conversation, not just answers a text query, with
call-specific telemetry (transcription confidence, interruptions,
turn-taking latency) that a text-only AI product never has to measure.

THREE DIFFERENT SIGNAL SHAPES ARE EMBEDDED, matching the multi-detector
pattern used elsewhere in this session:

1. DEPLOY-TIED REGIONAL DIALECT REGRESSION (the headline finding): a
   voice-model backbone upgrade (v1 -> v2) improves containment for
   every driver region EXCEPT Appalachian, where transcription
   confidence and containment both regress -- a regional-dialect
   coverage gap in the new model's training data, hidden in the
   nationwide aggregate because Appalachian is a minority share of call
   volume. This is a real, well-documented category of voice-AI risk
   (regional/dialectal ASR performance gaps), not a fabricated one.
2. TRANSIENT ASR OUTAGE (day 12, hours 10-14): a shared upstream speech-
   recognition provider issue crashes transcription confidence and
   spikes escalations across EVERY region and call type simultaneously
   -- an infra incident, not a model or fairness issue.
3. BENIGN DEPLOY (day 35): a shortened greeting script reduces average
   call duration by design, with no change to containment/success rate
   -- the false positive a naive "big metric change = bad" detector
   would flag, that a deploy-aware containment test correctly dismisses.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(1177)

START = pd.Timestamp("2026-05-01")
N_DAYS = 50
N_CALLS = 20000

DEPLOY_DAY = 25  # voice-model backbone v1 -> v2
DEPLOY_TS = START + pd.Timedelta(days=DEPLOY_DAY)
ASR_OUTAGE_START = START + pd.Timedelta(days=12, hours=10)
ASR_OUTAGE_END = START + pd.Timedelta(days=12, hours=14)
BENIGN_DEPLOY_DAY = 35  # shortened greeting script, no model change
BENIGN_DEPLOY_TS = START + pd.Timedelta(days=BENIGN_DEPLOY_DAY)

CALL_TYPES = ["eta_checkin", "exception_report", "pod_confirmation", "reschedule_request"]
CALL_TYPE_WEIGHT = {"eta_checkin": 0.40, "exception_report": 0.20, "pod_confirmation": 0.25, "reschedule_request": 0.15}

REGIONS = ["Midwest", "Gulf_Coast", "Northeast", "Pacific_Northwest", "Appalachian"]
REGION_WEIGHT = {"Midwest": 0.30, "Gulf_Coast": 0.20, "Northeast": 0.15, "Pacific_Northwest": 0.15, "Appalachian": 0.20}

# containment (call resolved without human dispatcher) by region and voice-model version
CONTAINMENT = {
    ("Midwest", "v1"): 0.78, ("Midwest", "v2"): 0.85,
    ("Gulf_Coast", "v1"): 0.75, ("Gulf_Coast", "v2"): 0.82,
    ("Northeast", "v1"): 0.80, ("Northeast", "v2"): 0.86,
    ("Pacific_Northwest", "v1"): 0.77, ("Pacific_Northwest", "v2"): 0.84,
    ("Appalachian", "v1"): 0.74, ("Appalachian", "v2"): 0.65,  # the hidden regional regression
}
# mean transcription confidence by region and version -- tracks containment closely
CONFIDENCE_MEAN = {
    ("Midwest", "v1"): 0.88, ("Midwest", "v2"): 0.93,
    ("Gulf_Coast", "v1"): 0.85, ("Gulf_Coast", "v2"): 0.90,
    ("Northeast", "v1"): 0.89, ("Northeast", "v2"): 0.94,
    ("Pacific_Northwest", "v1"): 0.86, ("Pacific_Northwest", "v2"): 0.91,
    ("Appalachian", "v1"): 0.82, ("Appalachian", "v2"): 0.71,
}
BASE_LATENCY_MS = 900
BASE_DURATION_SEC = {"eta_checkin": 55, "exception_report": 150, "pod_confirmation": 70, "reschedule_request": 110}


def business_hours_multiplier(hour: int) -> float:
    """Truck drivers are reachable mostly during daytime driving hours."""
    return 0.25 + 0.75 * np.exp(-0.5 * ((hour - 13) / 4.5) ** 2)


hours_of_day = np.arange(24)
hour_weights = business_hours_multiplier(hours_of_day)
day_offsets = rng.integers(0, N_DAYS, N_CALLS)
hour_choices = rng.choice(hours_of_day, size=N_CALLS, p=hour_weights / hour_weights.sum())
timestamps = START + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(hour_choices, unit="h") \
    + pd.to_timedelta(rng.integers(0, 3600, N_CALLS), unit="s")

rows = []
for i in range(N_CALLS):
    ts = timestamps[i]
    call_type = rng.choice(CALL_TYPES, p=[CALL_TYPE_WEIGHT[c] for c in CALL_TYPES])
    region = rng.choice(REGIONS, p=[REGION_WEIGHT[r] for r in REGIONS])
    version = "v2" if ts >= DEPLOY_TS else "v1"

    in_outage = ASR_OUTAGE_START <= ts < ASR_OUTAGE_END

    conf_mean = CONFIDENCE_MEAN[(region, version)]
    if in_outage:
        conf_mean = 0.38
    transcription_confidence = float(np.clip(rng.normal(conf_mean, 0.06), 0.05, 0.99))

    n_interruptions = max(0, int(round(rng.poisson(0.8 + (1 - transcription_confidence) * 4.5))))

    latency_target = BASE_LATENCY_MS
    if in_outage:
        latency_target *= 2.8
    avg_response_latency_ms = max(150, rng.normal(latency_target, latency_target * 0.15))

    duration_target = BASE_DURATION_SEC[call_type]
    if ts >= BENIGN_DEPLOY_TS:
        duration_target *= 0.83  # shortened greeting script -- planned, benign
    call_duration_sec = max(15, rng.normal(duration_target, duration_target * 0.25))

    p_contain = CONTAINMENT[(region, version)]
    if in_outage:
        p_contain *= 0.45  # ASR outage drives real escalations too, independent of region/model
    p_contain = float(np.clip(p_contain + rng.normal(0, 0.02), 0.05, 0.97))
    escalated_to_human = bool(rng.random() > p_contain)
    outcome_success = (not escalated_to_human) or bool(rng.random() < 0.9)  # escalated calls mostly still resolved via human

    driver_sentiment = int(np.clip(round(rng.normal(
        4.3 if (not escalated_to_human and not in_outage) else (3.0 if escalated_to_human else 2.2), 0.7)), 1, 5))

    rows.append({
        "call_id": f"C{900000+i}", "timestamp": ts, "call_type": call_type, "driver_region": region,
        "model_version": version, "call_duration_sec": round(call_duration_sec, 1),
        "transcription_confidence": round(transcription_confidence, 3), "n_interruptions": n_interruptions,
        "avg_response_latency_ms": round(avg_response_latency_ms, 1), "escalated_to_human": escalated_to_human,
        "outcome_success": outcome_success, "driver_sentiment_score": driver_sentiment,
    })

calls = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
calls.to_csv("calls.csv", index=False)

print(f"{len(calls):,} calls | {calls.timestamp.min()} to {calls.timestamp.max()} | "
      f"model deploy day {DEPLOY_DAY} | ASR outage day 12 (10:00-14:00) | benign deploy day {BENIGN_DEPLOY_DAY}")
for version in ["v1", "v2"]:
    sub = calls[(calls.model_version == version)]
    print(f"\n--- {version} (excluding outage window) ---")
    sub2 = sub[~((sub.timestamp >= ASR_OUTAGE_START) & (sub.timestamp < ASR_OUTAGE_END))]
    g = sub2.groupby("driver_region").agg(
        n=("call_id", "count"), containment=("escalated_to_human", lambda s: 1 - s.mean()),
        avg_confidence=("transcription_confidence", "mean"),
    ).round(4)
    print(g)
print("\nWrote calls.csv")
