import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

FILE_PATH = "/Users/antonsalonen/Desktop/DATASET1.xlsx"
OUTPUT_PATH = "/Users/antonsalonen/Desktop/OMXH25_LSTM_forecasts.csv"

# Model parameters
TARGET_COL = "Close"
FEATURE_COLS = [
    "Close",
    "VSTOXX",
    "EUR/USD",
    "ECB main refinancing rate",
    "Consumer confidence",
    "Unemployment rate"
]

SEQUENCE_LENGTH = 20
NEURONS = 150
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 1000
PATIENCE = 10
MIN_DELTA = 0.0001

TRAIN_RATIO = 0.80
VAL_RATIO_WITHIN_TRAIN = 0.10

df = pd.read_excel(FILE_PATH)
df = df.set_index("Date")

X_raw = df[FEATURE_COLS].to_numpy()
y_raw = df[[TARGET_COL]].to_numpy()

# Train / Val / Test split
n = len(X_raw)
trainval_end = int(TRAIN_RATIO * n)

X_trainval_raw = X_raw[:trainval_end]
X_test_raw = X_raw[trainval_end:]

y_trainval_raw = y_raw[:trainval_end]
y_test_raw = y_raw[trainval_end:]

# Validation set for early stopping is the last 10% of the training set
val_size = int(VAL_RATIO_WITHIN_TRAIN * len(X_trainval_raw))
fit_end = len(X_trainval_raw) - val_size

X_train_raw = X_trainval_raw[:fit_end]
X_val_raw = X_trainval_raw[fit_end:]

y_train_raw = y_trainval_raw[:fit_end]
y_val_raw = y_trainval_raw[fit_end:]

# Min-max normalization: fit on training set only as data after training set is considered to be the future
x_scaler = MinMaxScaler().fit(X_train_raw)
y_scaler = MinMaxScaler().fit(y_train_raw)

X_train = x_scaler.transform(X_train_raw)
X_val = x_scaler.transform(X_val_raw)
X_test = x_scaler.transform(X_test_raw)
X_trainval = x_scaler.transform(X_trainval_raw)

y_train = y_scaler.transform(y_train_raw)
y_val = y_scaler.transform(y_val_raw)
y_test = y_scaler.transform(y_test_raw)
y_trainval = y_scaler.transform(y_trainval_raw)

# Sequences for training
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_train,
    targets=y_train[SEQUENCE_LENGTH:],
    sequence_length=SEQUENCE_LENGTH,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Validation sequences, allow validation sequences to use last training observations to ensure there are no gaps
X_val_ext = np.concatenate([X_train[-SEQUENCE_LENGTH:], X_val], axis=0)

val_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_val_ext,
    targets=y_val,
    sequence_length=SEQUENCE_LENGTH,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Full pre-test training dataset for final retraining
trainval_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_trainval,
    targets=y_trainval[SEQUENCE_LENGTH:],
    sequence_length=SEQUENCE_LENGTH,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Testing sequences, allow test sequences to use last pre-test observations to ensure there are no gaps
X_test_ext = np.concatenate([X_trainval[-SEQUENCE_LENGTH:], X_test], axis=0)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_test_ext,
    targets=y_test,
    sequence_length=SEQUENCE_LENGTH,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# LSTM model
def build_model(seq_len, n_features):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(NEURONS),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adagrad(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model

# Stage 1 determine best number of epochs using validation data
model = build_model(SEQUENCE_LENGTH, X_train.shape[1])

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
print("Best epoch from validation:", best_epoch)

# Stage 2 retrain model on full training data, 80% of all data
final_model = build_model(SEQUENCE_LENGTH, X_train.shape[1])

final_model.fit(
    trainval_ds,
    epochs=best_epoch,
    verbose=1
)

# Make predictions on test data
y_pred_scaled = final_model.predict(test_ds, verbose=0)

# True target values from test_ds to ensure alignment with predictions
y_true_scaled = y_test

#Scaling back to real prices
y_true_price = y_scaler.inverse_transform(y_true_scaled)
y_pred_price = y_scaler.inverse_transform(y_pred_scaled)

# SHAP ANALYSIS
def ds_to_numpy(ds, max_batches=10):
    xs = []
    for i, (x, _) in enumerate(ds):
        if i >= max_batches:
            break
        xs.append(x.numpy())
    return np.concatenate(xs, axis=0)

# Background sample from full pre-test training data
X_bg = ds_to_numpy(trainval_ds, max_batches=10)

# Test sequences to explain
X_explain = ds_to_numpy(test_ds, max_batches=5)

# SHAP GradientExplainer for final model
explainer = shap.GradientExplainer(final_model, X_bg)
shap_values = explainer.shap_values(X_explain)

# Handle output format
if isinstance(shap_values, list):
    shap_values = shap_values[0]
# Checking for correct shape
if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
    shap_values = shap_values[..., 0]

# Output 1: SHAP Global feature importance
mean_abs_shap_feature = np.mean(np.abs(shap_values), axis=(0, 1))

shap_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean_abs_shap": mean_abs_shap_feature
}).sort_values("mean_abs_shap", ascending=False)

print("\n GLOBAL SHAP FEATURE IMPORTANCE")
print(shap_importance)

# Output 2 SHAP by LAG
mean_abs_shap_lag = np.mean(np.abs(shap_values), axis=(0, 2))

lag_labels = np.arange(1, SEQUENCE_LENGTH + 1)

lag_importance = pd.DataFrame({
    "lag": lag_labels,
    "mean_abs_shap": mean_abs_shap_lag
})

plt.figure(figsize=(10, 5))
plt.plot(lag_importance["lag"], lag_importance["mean_abs_shap"])
plt.xticks(np.arange(2,SEQUENCE_LENGTH +1 ,2))
plt.title("SHAP Importance by Lag")
plt.xlabel("Lag in Input Sequence")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.show()

# Output 3 SHAP heatmap per feature per lag
mean_abs_shap_lag_feature = np.mean(np.abs(shap_values), axis=0)

plt.figure(figsize=(12, 6))
plt.imshow(mean_abs_shap_lag_feature.T, aspect="auto", cmap="magma")
plt.colorbar(label="Mean |SHAP value|")
plt.yticks(range(len(FEATURE_COLS)), FEATURE_COLS)

tick_positions = np.linspace(0, SEQUENCE_LENGTH - 1, 10, dtype=int)
tick_labels = [str(i + 1) for i in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.title("SHAP Heatmap: Feature Importance Across Lags")
plt.xlabel("Lag in Input Sequence")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Output 4 SHAP plot
plt.figure(figsize=(10, 6))

# Normalize lag values for coloring
n_lags = shap_values.shape[1]
lag_indices = np.arange(n_lags)

for i, feature in enumerate(FEATURE_COLS):
    # Collect SHAP values and corresponding lag indices
    shap_feat = shap_values[:, :, i]  # (samples, time_steps)

    y = shap_feat.flatten()
    # Repeat lag indices for all samples
    lag_rep = np.tile(lag_indices, shap_feat.shape[0])
    # X positions with jitter
    x = i + np.random.normal(0, 0.08, size=len(y))
    sc = plt.scatter(x, y, c=lag_rep, cmap='magma', alpha=0.3, s=10)

# Colorbar for lag
cbar = plt.colorbar(sc)
cbar.set_label("Lag (time step)")
cbar.set_ticks(np.arange(2, SEQUENCE_LENGTH + 1, 2))

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xticks(range(len(FEATURE_COLS)), FEATURE_COLS, rotation=45)
plt.ylabel("SHAP value")
plt.xlabel("Feature")
plt.title("SHAP Value Distribution by Feature (Colored by Lag)")

plt.tight_layout()
plt.show()

# Export forecasts
actual_prices = y_true_price.flatten()
forecast_prices = y_pred_price.flatten()

test_dates = df.index[trainval_end:]

forecast_df = pd.DataFrame({
    "Actual_Close": actual_prices,
    "Forecast_Close": forecast_prices
}, index=test_dates)

forecast_df.to_csv(OUTPUT_PATH)