"""
CASE 05 — Demand Forecasting & Markdown ROI (Retail)
Real Walmart historical sales data (train.csv) + external regressors
(features.csv) — Store 20, all departments aggregated to a weekly series.
Holt-Winters exponential smoothing vs. a seasonal-naive baseline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams["font.family"] = "DejaVu Sans"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; RED="#e34948"; GOOD="#0ca30c"
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; SURFACE="#fcfcfb"

STORE = 20

train = pd.read_csv("../data/train.csv", parse_dates=["Date"])
feat = pd.read_csv("../data/features.csv", parse_dates=["Date"])

s = train[train["Store"] == STORE].groupby("Date")["Weekly_Sales"].sum().reset_index()
s = s.sort_values("Date").reset_index(drop=True)

f = feat[feat["Store"] == STORE].copy()
md_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
f["MarkDown_total"] = f[md_cols].fillna(0).sum(axis=1)
f = f[["Date","IsHoliday","Temperature","Fuel_Price","CPI","Unemployment","MarkDown_total"]]

s = s.merge(f, on="Date", how="left")
s = s.set_index("Date")
s.index.freq = "W-FRI"

N_TEST = 16
train_s = s.iloc[:-N_TEST]
test_s = s.iloc[-N_TEST:]

# ---- Baseline: seasonal-naive (value from 52 weeks earlier) ----
lag = 52
baseline_pred = []
full_sales = s["Weekly_Sales"]
for date in test_s.index:
    ref_date = date - pd.Timedelta(weeks=lag)
    if ref_date in full_sales.index:
        baseline_pred.append(full_sales.loc[ref_date])
    else:
        baseline_pred.append(np.nan)
baseline_pred = pd.Series(baseline_pred, index=test_s.index)

# ---- Holt-Winters (additive trend + additive seasonal, period=52) ----
hw_model = ExponentialSmoothing(
    train_s["Weekly_Sales"], trend="add", seasonal="add", seasonal_periods=52,
    initialization_method="estimated"
).fit()
hw_pred = hw_model.forecast(N_TEST)
hw_pred.index = test_s.index

def mape(actual, pred):
    mask = ~pred.isna()
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def rmse(actual, pred):
    mask = ~pred.isna()
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))

results = {
    "store": STORE,
    "n_weeks_total": int(len(s)),
    "n_train": int(len(train_s)),
    "n_test": N_TEST,
    "baseline_mape": round(mape(test_s["Weekly_Sales"], baseline_pred), 2),
    "baseline_rmse": round(rmse(test_s["Weekly_Sales"], baseline_pred), 0),
    "hw_mape": round(mape(test_s["Weekly_Sales"], hw_pred), 2),
    "hw_rmse": round(rmse(test_s["Weekly_Sales"], hw_pred), 0),
}
results["improvement_pct"] = round(100 * (results["baseline_mape"] - results["hw_mape"]) / results["baseline_mape"], 1)

# ---------------------------------------------------------------
# Markdown / holiday lift analysis (statistical test)
# ---------------------------------------------------------------
s_reset = s.reset_index()
holiday_sales = s_reset.loc[s_reset["IsHoliday"] == True, "Weekly_Sales"]
normal_sales = s_reset.loc[s_reset["IsHoliday"] == False, "Weekly_Sales"]
t_holiday, p_holiday = stats.ttest_ind(holiday_sales, normal_sales, equal_var=False)

md_active = s_reset.loc[s_reset["MarkDown_total"] > 1000, "Weekly_Sales"]
md_inactive = s_reset.loc[s_reset["MarkDown_total"] <= 1000, "Weekly_Sales"]
t_md, p_md = stats.ttest_ind(md_active, md_inactive, equal_var=False)

results["holiday_avg"] = round(float(holiday_sales.mean()), 0)
results["normal_avg"] = round(float(normal_sales.mean()), 0)
results["holiday_lift_pct"] = round(100 * (holiday_sales.mean() - normal_sales.mean()) / normal_sales.mean(), 1)
results["holiday_p_value"] = round(float(p_holiday), 4)

results["markdown_active_avg"] = round(float(md_active.mean()), 0)
results["markdown_inactive_avg"] = round(float(md_inactive.mean()), 0)
results["markdown_lift_pct"] = round(100 * (md_active.mean() - md_inactive.mean()) / md_inactive.mean(), 1)
results["markdown_p_value"] = round(float(p_md), 4)
results["markdown_n_active_weeks"] = int(len(md_active))

with open("forecast_results.json", "w") as f_out:
    json.dump(results, f_out, indent=2)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------
# CHART 1 — Forecast vs actual (test period)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
context = s.iloc[-(N_TEST+20):]
ax.plot(context.index, context["Weekly_Sales"], color=INK, linewidth=1.6, label="Actual", zorder=4)
ax.plot(test_s.index, baseline_pred, color=MUTED, linewidth=1.8, linestyle="--", label=f"Seasonal-naive (MAPE {results['baseline_mape']:.1f}%)", zorder=3)
ax.plot(test_s.index, hw_pred, color=BLUE, linewidth=2.2, label=f"Holt-Winters (MAPE {results['hw_mape']:.1f}%)", zorder=3)
ax.axvline(test_s.index[0], color=RED, linewidth=1, linestyle=":", alpha=0.7)
ax.text(test_s.index[0], ax.get_ylim()[1] if ax.get_ylim()[1] else 0, "", fontsize=8)
ax.set_title(f"Store {STORE}: Holt-Winters beats naive by {results['improvement_pct']:.0f}%", fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.spines[["left","bottom"]].set_color(GRID)
ax.tick_params(labelsize=8.5, colors=MUTED)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(color=GRID, linewidth=0.7, zorder=0)
ax.legend(frameon=False, fontsize=8.7, loc="upper left", labelcolor=INK2)
plt.tight_layout()
plt.savefig("chart_forecast_vs_actual.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 2 — Seasonal decomposition
# ---------------------------------------------------------------
decomp = seasonal_decompose(s["Weekly_Sales"], model="additive", period=52, extrapolate_trend="freq")
fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), dpi=200, sharex=True)
fig.patch.set_facecolor(SURFACE)
parts = [("Trend", decomp.trend, BLUE), ("Seasonal", decomp.seasonal, AQUA), ("Residual", decomp.resid, MUTED)]
for ax, (name, series, color) in zip(axes, parts):
    ax.set_facecolor(SURFACE)
    ax.plot(series.index, series.values, color=color, linewidth=1.5)
    ax.set_ylabel(name, fontsize=9.5, color=INK2)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
axes[0].set_title(f"Store {STORE} weekly sales — decomposed", fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout()
plt.savefig("chart_seasonal_decomposition.png", facecolor=SURFACE)
plt.close()

# ---------------------------------------------------------------
# CHART 3 — Holiday & markdown lift
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
cats = ["Non-holiday\nweeks", "Holiday\nweeks", "Markdown-\ninactive weeks", "Markdown-\nactive weeks"]
vals = [results["normal_avg"], results["holiday_avg"], results["markdown_inactive_avg"], results["markdown_active_avg"]]
colors = [MUTED, RED, MUTED, GOOD]
bars = ax.bar(cats, vals, color=colors, width=0.55, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v + max(vals)*0.02, f"${v/1e6:.2f}M", ha="center", fontsize=9.5, color=INK, fontweight="bold")
ax.set_title("Both holidays and markdowns lift weekly sales", fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=8.7, colors=INK2)
ax.tick_params(axis="y", length=0, labelsize=8.5, colors=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
plt.tight_layout()
plt.savefig("chart_holiday_markdown_lift.png", facecolor=SURFACE)
plt.close()

print("\nCharts saved.")
print(f"Holiday lift: {results['holiday_lift_pct']}% (p={results['holiday_p_value']})")
print(f"Markdown lift: {results['markdown_lift_pct']}% (p={results['markdown_p_value']}, n_active_weeks={results['markdown_n_active_weeks']})")

# Save forecast + errors for the Excel replenishment planner
forecast_export = pd.DataFrame({
    "Date": test_s.index, "Actual": test_s["Weekly_Sales"].values,
    "Forecast_HoltWinters": hw_pred.values, "Forecast_Baseline": baseline_pred.values,
})
forecast_export["Abs_Error_HW"] = (forecast_export["Actual"] - forecast_export["Forecast_HoltWinters"]).abs()
forecast_export.to_csv("forecast_test_period.csv", index=False)
print("Forecast errors exported for planner workbook. Std dev of HW forecast error:", forecast_export["Abs_Error_HW"].std())
