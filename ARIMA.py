import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

FILE_PATH = "/Users/antonsalonen/Desktop/DATASET2.xlsx"
EXPORT_PATH = "/Users/antonsalonen/Desktop/ARIMA_111_forecasts.csv"

df = pd.read_excel(FILE_PATH)
df = df.set_index("Date")

# log prices
y = np.log(df["Close"].astype(float)).dropna()
y.name = "log_close"

# Train/test splitting
split_idx = int(len(y) * 0.80)
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Box–Jenkins methodology, differencing for stationarity
y_train_diff = y_train.diff().dropna()

plt.figure(figsize=(10, 4))
plot_acf(y_train_diff, lags=40)
plt.title("ACF of Differenced Log Price")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(y_train_diff, lags=40)
plt.title("PACF of Differenced Log Price")
plt.show()

# Estimate ARIMA(0,1,0) candidates on training data no drift and drift
candidate_nodrift = ARIMA(y_train, order=(0, 1, 0), trend="n").fit()
candidate_drift = ARIMA(y_train, order=(0, 1, 0), trend="t").fit()

print("\nARIMA(0,1,0) no drift (TRAIN):")
print(candidate_nodrift.summary())

print("\nARIMA(0,1,0) with drift (TRAIN):")
print(candidate_drift.summary())

# Model selection with AIC and BIC
print("\nModel comparison:")
print(f"ARIMA(0,1,0) no drift    AIC: {candidate_nodrift.aic:.3f}   BIC: {candidate_nodrift.bic:.3f}")
print(f"ARIMA(0,1,0) with drift  AIC: {candidate_drift.aic:.3f}   BIC: {candidate_drift.bic:.3f}")

# Choose the benchmark model using AIC and BIC. If both favor drift, choose drift, otherwise choose no drift due to parsimony.
if (candidate_drift.aic < candidate_nodrift.aic) and (candidate_drift.bic < candidate_nodrift.bic):
    chosen_model = candidate_drift
    chosen_name = "ARIMA(0,1,0) with drift (trend='t')"
else:
    chosen_model = candidate_nodrift
    chosen_name = "ARIMA(0,1,0) no drift (trend='n')"

print("\nChosen benchmark model:", chosen_name)

# Residual diagnostics for the chosen model
residuals = chosen_model.resid.dropna()

lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\nLjung–Box test (chosen model residuals):")
print(lb_test)

# ARIMA(1,1,1) added separately for trading benchmark purposes
history = y_train.copy()
arima_111_forecasts = []

# Warnings regarding inferred time index frequency were suppressed to not drown the terminal, as they are informational and do not impact the model estimation or forecasting outcomes
import warnings
warnings.filterwarnings(
    "ignore",
    message="No frequency information was provided, so inferred frequency"
)

for t in range(len(y_test)):
    model = ARIMA(history, order=(1, 1, 1), trend="n")
    fitted = model.fit()
    forecast = fitted.forecast(steps=1).iloc[0]
    arima_111_forecasts.append(forecast)
    history = pd.concat([history, pd.Series([y_test.iloc[t]], index=[y_test.index[t]])])

# Store forecasts as series aligned with test index
arima_111_forecasts = pd.Series(arima_111_forecasts, index=y_test.index, name="ARIMA_111_Forecast")

# Convert from log prices back to actual prices
arima_111_price_forecasts = np.exp(arima_111_forecasts)
arima_111_price_forecasts.name = "ARIMA_111_Forecast"

arima_111_price_forecasts.to_csv(EXPORT_PATH, header=True)