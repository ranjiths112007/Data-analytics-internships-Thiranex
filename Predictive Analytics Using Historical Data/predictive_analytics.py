"""
Predictive Analytics — Air Passenger Forecasting
=================================================
Dataset : AirPassengers.csv  (place in same folder)
Models  : Linear Regression · Random Forest · SARIMA
Run     : python predictive_analytics.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_FILE      = "AirPassengers.csv"
TEST_MONTHS    = 24
FORECAST_MONTHS = 12
# ─────────────────────────────────────────────────────────────────────


# 1. LOAD & CLEAN
df = pd.read_csv(DATA_FILE)
df.columns = ["date", "passengers"]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
df.dropna(inplace=True)

print(f"✅  Loaded {len(df)} rows  |  {df['date'].min().date()} → {df['date'].max().date()}")
print(f"    Passengers range: {df['passengers'].min()} – {df['passengers'].max()}")


# 2. FEATURE ENGINEERING
df["trend"]   = np.arange(len(df))
df["month"]   = df["date"].dt.month
df["lag_1"]   = df["passengers"].shift(1)
df["lag_12"]  = df["passengers"].shift(12)
df["roll_12"] = df["passengers"].shift(1).rolling(12).mean()
df.dropna(inplace=True)

FEATURES = ["trend", "month", "lag_1", "lag_12", "roll_12"]
split    = len(df) - TEST_MONTHS
train    = df.iloc[:split]
test     = df.iloc[split:]
X_tr, y_tr = train[FEATURES], train["passengers"]
X_te, y_te = test[FEATURES],  test["passengers"]


# 3. TRAIN MODELS
lr = LinearRegression().fit(X_tr, y_tr)
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)

sarima = SARIMAX(
    train["passengers"], order=(1,1,1), seasonal_order=(1,1,1,12),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)


# 4. EVALUATE
def metrics(name, actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2   = r2_score(actual, predicted)
    print(f"  {name:<20}  MAE={mae:6.1f}  MAPE={mape:5.1f}%  R²={r2:.3f}")
    return predicted

print("\n📊 Model Evaluation (test set):")
lr_pred     = metrics("Linear Regression", y_te.values, lr.predict(X_te))
rf_pred     = metrics("Random Forest",     y_te.values, rf.predict(X_te))
sarima_pred = metrics("SARIMA",            y_te.values, sarima.forecast(TEST_MONTHS).values)


# 5. FUTURE FORECAST
full_sarima = SARIMAX(
    df["passengers"], order=(1,1,1), seasonal_order=(1,1,1,12),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)

fc           = full_sarima.get_forecast(steps=FORECAST_MONTHS)
fc_mean      = fc.predicted_mean
fc_ci        = fc.conf_int(alpha=0.20)
future_dates = pd.date_range(
    df["date"].max() + pd.DateOffset(months=1), periods=FORECAST_MONTHS, freq="MS"
)


# 6. DASHBOARD
BG, CARD, LINE = "#0d1117", "#161b22", "#21262d"
C = {"actual":"#58a6ff", "lr":"#bc8cff", "rf":"#3fb950",
     "sarima":"#ffa657", "forecast":"#f85149", "ci":"#f85149"}

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle("Air Passenger Forecasting — Predictive Analytics",
             fontsize=20, fontweight="bold", color="white", y=0.97)
gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.50, wspace=0.32,
                       left=0.06, right=0.97, top=0.92, bottom=0.05)

def ax_style(ax, title="", xl="", yl=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for sp in ax.spines.values(): sp.set_color(LINE)
    if title: ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    if xl:    ax.set_xlabel(xl,    color="#8b949e", fontsize=9)
    if yl:    ax.set_ylabel(yl,    color="#8b949e", fontsize=9)

# Panel 1 — full timeline + forecast
ax1 = fig.add_subplot(gs[0, :])
ax_style(ax1, "Historical Data + 12-Month Forecast (SARIMA)", "Date", "Passengers")
ax1.plot(df["date"], df["passengers"], color=C["actual"], lw=1.8, label="Actual")
ax1.plot(future_dates, fc_mean, color=C["forecast"], lw=2.2, ls="--", label="Forecast")
ax1.fill_between(future_dates, fc_ci.iloc[:,0], fc_ci.iloc[:,1],
                 color=C["ci"], alpha=0.15, label="80% CI")
ax1.axvline(df["date"].iloc[split], color="#8b949e", ls=":", lw=1)
ax1.text(df["date"].iloc[split], ax1.get_ylim()[0], " test split",
         color="#8b949e", fontsize=8, va="bottom")
ax1.legend(framealpha=0.2, facecolor=CARD, edgecolor=LINE, labelcolor="white", fontsize=9)

# Panel 2 — model comparison on test
ax2 = fig.add_subplot(gs[1, :2])
ax_style(ax2, "Model Predictions vs Actual (Test Set)", "Date", "Passengers")
ax2.plot(test["date"], y_te,        color=C["actual"], lw=2,   label="Actual")
ax2.plot(test["date"], lr_pred,     color=C["lr"],     lw=1.5, ls="--", label="Linear Regression")
ax2.plot(test["date"], rf_pred,     color=C["rf"],     lw=1.5, ls="--", label="Random Forest")
ax2.plot(test["date"], sarima_pred, color=C["sarima"], lw=1.5, ls="--", label="SARIMA")
ax2.legend(framealpha=0.2, facecolor=CARD, edgecolor=LINE, labelcolor="white", fontsize=9)

# Panel 3 — MAPE bar
ax3 = fig.add_subplot(gs[1, 2])
ax_style(ax3, "MAPE by Model (%)", "", "MAPE %")
mapes  = [
    np.mean(np.abs((y_te.values - lr_pred)     / y_te.values)) * 100,
    np.mean(np.abs((y_te.values - rf_pred)     / y_te.values)) * 100,
    np.mean(np.abs((y_te.values - sarima_pred) / y_te.values)) * 100,
]
bars = ax3.bar(["Linear\nRegression","Random\nForest","SARIMA"],
               mapes, color=[C["lr"],C["rf"],C["sarima"]], edgecolor=LINE, width=0.55)
for b, v in zip(bars, mapes):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
             f"{v:.1f}%", ha="center", color="white", fontsize=10, fontweight="bold")
ax3.set_ylim(0, max(mapes) * 1.3)

# Panel 4 — Actual vs Predicted scatter
ax4 = fig.add_subplot(gs[2, 0])
ax_style(ax4, "Actual vs Predicted (SARIMA)", "Actual", "Predicted")
ax4.scatter(y_te, sarima_pred, color=C["sarima"], s=55, alpha=0.85, edgecolors=LINE)
mn, mx = min(y_te.min(), sarima_pred.min()), max(y_te.max(), sarima_pred.max())
ax4.plot([mn,mx],[mn,mx], color="#8b949e", ls="--", lw=1)

# Panel 5 — Residuals
ax5 = fig.add_subplot(gs[2, 1])
ax_style(ax5, "Residuals (SARIMA)", "Predicted", "Error")
ax5.scatter(sarima_pred, y_te.values - sarima_pred,
            color=C["rf"], s=55, alpha=0.85, edgecolors=LINE)
ax5.axhline(0, color="#8b949e", ls="--", lw=1)

# Panel 6 — Forecast bar
ax6 = fig.add_subplot(gs[2, 2])
ax_style(ax6, "12-Month Forecast", "Month", "Passengers")
ax6.bar(future_dates.strftime("%b %y"), fc_mean.values,
        color=C["forecast"], edgecolor=LINE, width=0.65, alpha=0.9)
ax6.set_xticklabels(future_dates.strftime("%b %y"), rotation=45, ha="right",
                    color="#8b949e", fontsize=8)

plt.savefig("predictive_analytics_output.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print("\n✅  Dashboard saved → predictive_analytics_output.png")
print(f"\n🔮  12-Month Forecast:")
for d, v in zip(future_dates, fc_mean):
    print(f"    {d.strftime('%b %Y')}  →  {v:.0f} passengers")
