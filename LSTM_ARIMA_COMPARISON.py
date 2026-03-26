import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

LSTM_PATH = "/Users/antonsalonen/Desktop/OMXH25_LSTM_forecasts.csv"
ARIMA111_PATH = "/Users/antonsalonen/Desktop/ARIMA_111_forecasts.csv"
OUTPUT_DATA_PATH = "/Users/antonsalonen/Desktop/OMXH25_LSTM_vs_ARIMA_benchmarks.csv"
OUTPUT_METRICS_PATH = "/Users/antonsalonen/Desktop/forecast_metrics_comparison.csv"

df = pd.read_csv(LSTM_PATH, index_col=0, parse_dates=True).rename(
    columns={"Forecast_Close": "LSTM_Forecast"}
    )
arima111 = pd.read_csv(ARIMA111_PATH, index_col=0, parse_dates=True)

# Merge on date index
df = df.join(arima111[["ARIMA_111_Forecast"]], how="left")

# ARIMA(0,1,0)
df["ARIMA_010_Forecast"] = df["Actual_Close"].shift(1)

# Keep only rows where all forecasts exist
df = df.dropna(subset=["Actual_Close", "LSTM_Forecast", "ARIMA_010_Forecast", "ARIMA_111_Forecast"])

actual = df["Actual_Close"]

#Error metrics/diebold–mariano
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def diebold_mariano_test(y_true, pred1, pred2, h=1, power=2):
    y_true, pred1, pred2 = map(np.asarray, (y_true, pred1, pred2))
    d = np.abs(y_true - pred1) ** power - np.abs(y_true - pred2) ** power
    T = len(d)
    dm_stat = d.mean() / np.sqrt(d.var(ddof=1) / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat *= hln
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1))
    return dm_stat, p_value

models = {
    "LSTM": "LSTM_Forecast",
    "ARIMA(0,1,0)": "ARIMA_010_Forecast",
    "ARIMA(1,1,1)": "ARIMA_111_Forecast"
}

metrics_df = pd.DataFrame([
    {
        "Model": name,
        "RMSE": rmse(actual, df[col]),
        "MAPE (%)": mape(actual, df[col]),
        "Correlation (R)": actual.corr(df[col]),
    }
    for name, col in models.items()
])

print(metrics_df.to_string(index=False))

# Diebold-Mariano tests for LSTM against benchmark models
comparisons = [
    ("LSTM vs ARIMA(0,1,0)", "LSTM_Forecast", "ARIMA_010_Forecast"),
    ("LSTM vs ARIMA(1,1,1)", "LSTM_Forecast", "ARIMA_111_Forecast"),
]

dm_results = []

for label, col1, col2 in comparisons:
    dm_stat, dm_pvalue = diebold_mariano_test(actual, df[col1], df[col2], h=1, power=2)
    dm_results.append({
        "Comparison": label,
        "DM Statistic": dm_stat,
        "p-value": dm_pvalue,
        "Decision": (
            "Reject null hypothesis of equal predictive accuracy"
            if dm_pvalue < 0.05
            else "Cannot reject null hypothesis of equal predictive accuracy"
        )
    })

    print(f"\nDiebold-Mariano test ({label})")
    print(f"DM statistic: {dm_stat:.6f}")
    print(f"p-value: {dm_pvalue:.6f}")
    print(
        "Result: Reject null hypothesis of equal predictive accuracy."
        if dm_pvalue < 0.05
        else "Result: Cannot reject null hypothesis of equal predictive accuracy."
    )

dm_results_df = pd.DataFrame(dm_results)

df.to_csv(OUTPUT_DATA_PATH)
metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)

print(f"\nSaved comparison data to: {OUTPUT_DATA_PATH}")
print(f"Saved metrics table to: {OUTPUT_METRICS_PATH}")

#Plot forecasts vs actual
plt.figure(figsize=(12, 6))
plt.plot(df.index, actual, label="Actual Close")
plt.plot(df.index, df["LSTM_Forecast"], "--", label="LSTM Forecast")
plt.plot(df.index, df["ARIMA_010_Forecast"], ":", label="ARIMA(0,1,0) Forecast")
plt.plot(df.index, df["ARIMA_111_Forecast"], "-.", label="ARIMA(1,1,1) Forecast")
plt.title("Actual vs LSTM vs ARIMA Benchmarks")
plt.xlabel("Date")
plt.ylabel("OMXH25 Closing Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()