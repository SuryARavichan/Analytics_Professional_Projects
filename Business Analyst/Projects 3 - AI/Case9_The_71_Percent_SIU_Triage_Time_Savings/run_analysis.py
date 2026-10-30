"""
CASE 09 -- SIU Fraud-Risk Triage Model
Real historical auto insurance claims data (1,000 claims, widely-used teaching
dataset derived from real claims records). Logistic Regression vs. Random
Forest, evaluated for a claims-fraud triage/prioritization use case.

NOTE on base rate: this sample's fraud_reported rate is ~24.7%, far above
real-world SIU fraud prevalence (industry estimates are typically low single
digits). Treat this as a curated teaching/evaluation sample, not a
population-representative base rate -- see the PDF report for the full caveat.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, recall_score,
                              f1_score, confusion_matrix, accuracy_score)

plt.rcParams["font.family"] = "DejaVu Sans"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; RED="#e34948"; GOOD="#0ca30c"
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; SURFACE="#fcfcfb"

df = pd.read_csv("../data/insurance_claims.csv")
df = df.drop(columns=["_c39"])  # junk all-null column

# Identifiers, free-text, and near-unique fields dropped (no generalizable
# signal -- policy_number is an id, insured_zip/incident_location are
# effectively unique per row, dates are represented by months_as_customer /
# incident_hour_of_the_day already, auto_model has 39 near-unique levels on
# top of the auto_make family it belongs to).
id_like_cols = ["policy_number", "policy_bind_date", "insured_zip",
                 "incident_date", "incident_location", "auto_model"]
df_model = df.drop(columns=id_like_cols)

# "?" is this dataset's own missing-value marker for a handful of fields
# (collision_type, property_damage, police_report_available) -- recode to a
# real, explicit "UNKNOWN" category rather than silently one-hotting the
# literal string "?". Whether a field is unknown/unreported is itself a
# potentially meaningful signal for an investigator, so it is kept as its
# own category rather than imputed away.
df_model = df_model.replace("?", "UNKNOWN")
df_model["authorities_contacted"] = df_model["authorities_contacted"].fillna("None")

y = (df_model["fraud_reported"] == "Y").astype(int)
X_raw = df_model.drop(columns=["fraud_reported"])

categorical_cols = X_raw.select_dtypes(include="object").columns.tolist()
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)

# Keep a row-index-aligned reference frame (original, human-readable fields)
# for later use building the LLM investigator briefs and the Excel triage
# table, without polluting the model's feature matrix.
ref = df.loc[X.index].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
ref_test = ref.loc[X_test.index]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ---- Logistic Regression ----
logit = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
logit.fit(X_train_sc, y_train)
logit_proba = logit.predict_proba(X_test_sc)[:, 1]
logit_pred = (logit_proba >= 0.5).astype(int)

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= 0.5).astype(int)

def metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }

results = {
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(X.shape[1]),
    "fraud_rate": round(float(y.mean()), 3),
    "logit": metrics(y_test, logit_pred, logit_proba),
    "rf": metrics(y_test, rf_pred, rf_proba),
}

# SIU wants recall -- catching fraud matters more than avoiding a false flag,
# since every flag still goes to a human investigator before any decision.
chosen_name = "Random Forest" if results["rf"]["recall"] >= results["logit"]["recall"] else "Logistic Regression"
chosen_proba = rf_proba if chosen_name == "Random Forest" else logit_proba
chosen_pred = rf_pred if chosen_name == "Random Forest" else logit_pred
results["chosen_model"] = chosen_name

with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------
# CHART 1 -- ROC curves
# ---------------------------------------------------------------
fpr_l, tpr_l, _ = roc_curve(y_test, logit_proba)
fpr_r, tpr_r, _ = roc_curve(y_test, rf_proba)
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.plot(fpr_l, tpr_l, color=BLUE, linewidth=2.2, label=f"Logistic Regression (AUC {results['logit']['roc_auc']:.2f})")
ax.plot(fpr_r, tpr_r, color=ORANGE, linewidth=2.2, label=f"Random Forest (AUC {results['rf']['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", label="Random guess")
ax.set_xlabel("False positive rate", fontsize=10, color=INK2)
ax.set_ylabel("True positive rate (recall)", fontsize=10, color=INK2)
ax.set_title("Both models clear the random-guess baseline", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(GRID)
ax.tick_params(labelsize=9, colors=MUTED)
ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK2)
ax.grid(color=GRID, linewidth=0.7, zorder=0)
plt.tight_layout()
plt.savefig("chart_roc_curve.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 -- Feature importance (Random Forest, top 12)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
importances = importances.sort_values(ascending=True)
colors_bars = [RED if i == len(importances) - 1 else MUTED for i in range(len(importances))]
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.barh(importances.index, importances.values, color=colors_bars, height=0.62, zorder=3)
for b, v in zip(bars, importances.values):
    ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.8, color=INK)
ax.set_title("Incident severity dominates the risk signal", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("chart_feature_importance.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 3 -- Confusion matrix (chosen model)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, chosen_pred)
fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
im = ax.imshow(cm, cmap="Oranges", alpha=0.85)
labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center", fontsize=10.5,
                 color=INK if cm[i, j] < cm.max() * 0.6 else "white", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted:\nNot fraud", "Predicted:\nFraud"], fontsize=9.5, color=INK2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual:\nNot fraud", "Actual:\nFraud"], fontsize=9.5, color=INK2)
ax.set_title(f"{chosen_name} -- confusion matrix\n(test set, n={len(y_test)})", fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("chart_confusion_matrix.png", facecolor=SURFACE)
plt.close()

results["chosen_confusion_matrix"] = cm.tolist()
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------
# Persist the test-set scored claims (chosen model) for the Excel workbook
# and the LLM investigator-brief step -- real structured fields + probability.
# ---------------------------------------------------------------
scored = ref_test.copy()
scored["fraud_probability"] = chosen_proba
scored["actual_fraud_reported"] = ref_test["fraud_reported"]
scored = scored.sort_values("fraud_probability", ascending=False)
scored.to_csv("scored_test_claims.csv", index=False)

print("\nChosen model:", chosen_name)
print("Confusion matrix:", cm.tolist())
print("Charts + scored_test_claims.csv saved.")
