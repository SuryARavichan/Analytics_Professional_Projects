"""
CASE 04 — Predictive Risk Screening for Diabetes Care Management
Real Pima Indians Diabetes Database (National Institute of Diabetes and
Digestive and Kidney Diseases, 768 patients). Logistic Regression vs
Random Forest, evaluated for a proactive outreach use case.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, recall_score,
                              f1_score, confusion_matrix, accuracy_score)

plt.rcParams["font.family"] = "DejaVu Sans"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; RED="#e34948"; GOOD="#0ca30c"
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; SURFACE="#fcfcfb"

cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
        "DiabetesPedigreeFunction","Age","Outcome"]
df = pd.read_csv("../data/pima_diabetes.csv", header=None, names=cols)

# Biologically implausible zeros -> missing (documented data-quality step)
zero_as_missing = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for c in zero_as_missing:
    df[c] = df[c].replace(0, np.nan)

missing_summary = df[zero_as_missing].isna().sum().to_dict()

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)

# ---- Logistic Regression ----
logit = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logit.fit(X_train_sc, y_train)
logit_proba = logit.predict_proba(X_test_sc)[:, 1]
logit_pred = (logit_proba >= 0.5).astype(int)

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X_train_imp, y_train)
rf_proba = rf.predict_proba(X_test_imp)[:, 1]
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
    "positive_rate": round(float(y.mean()), 3),
    "missing_zeros": {k: int(v) for k, v in missing_summary.items()},
    "logit": metrics(y_test, logit_pred, logit_proba),
    "rf": metrics(y_test, rf_pred, rf_proba),
}
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------
# CHART 1 — ROC curves
# ---------------------------------------------------------------
fpr_l, tpr_l, _ = roc_curve(y_test, logit_proba)
fpr_r, tpr_r, _ = roc_curve(y_test, rf_proba)
fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.plot(fpr_l, tpr_l, color=BLUE, linewidth=2.2, label=f"Logistic Regression (AUC {results['logit']['roc_auc']:.2f})")
ax.plot(fpr_r, tpr_r, color=ORANGE, linewidth=2.2, label=f"Random Forest (AUC {results['rf']['roc_auc']:.2f})")
ax.plot([0,1],[0,1], color=MUTED, linewidth=1.2, linestyle="--", label="Random guess")
ax.set_xlabel("False positive rate", fontsize=10, color=INK2)
ax.set_ylabel("True positive rate (recall)", fontsize=10, color=INK2)
ax.set_title("Both models beat chance comfortably", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.spines[["left","bottom"]].set_color(GRID)
ax.tick_params(labelsize=9, colors=MUTED)
ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK2)
ax.grid(color=GRID, linewidth=0.7, zorder=0)
plt.tight_layout()
plt.savefig("chart_roc_curve.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — Feature importance (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
colors = [RED if i == len(importances)-1 else MUTED for i in range(len(importances))]
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bars = ax.barh(importances.index, importances.values, color=colors, height=0.6, zorder=3)
for b, v in zip(bars, importances.values):
    ax.text(v + 0.005, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=9, color=INK)
ax.set_title("Glucose dominates the risk signal", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK2)
ax.tick_params(axis="x", length=0, labelsize=9, colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig("chart_feature_importance.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 3 — Confusion matrix (chosen model: Logistic Regression, higher recall priority)
# ---------------------------------------------------------------
chosen_pred = logit_pred if results["logit"]["recall"] >= results["rf"]["recall"] else rf_pred
chosen_name = "Logistic Regression" if results["logit"]["recall"] >= results["rf"]["recall"] else "Random Forest"
cm = confusion_matrix(y_test, chosen_pred)
fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
im = ax.imshow(cm, cmap="Oranges", alpha=0.85)
labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}\n{labels[i][j]}", ha="center", va="center", fontsize=10.5,
                color=INK if cm[i,j] < cm.max()*0.6 else "white", fontweight="bold")
ax.set_xticks([0,1]); ax.set_xticklabels(["Predicted:\nNo diabetes","Predicted:\nDiabetes"], fontsize=9.5, color=INK2)
ax.set_yticks([0,1]); ax.set_yticklabels(["Actual:\nNo diabetes","Actual:\nDiabetes"], fontsize=9.5, color=INK2)
ax.set_title(f"{chosen_name} — confusion matrix\n(test set, n={len(y_test)})", fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("chart_confusion_matrix.png", facecolor=SURFACE)
plt.close()

results["chosen_model"] = chosen_name
results["chosen_confusion_matrix"] = cm.tolist()
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nChosen model:", chosen_name)
print("Confusion matrix:", cm.tolist())
print("Charts saved.")
