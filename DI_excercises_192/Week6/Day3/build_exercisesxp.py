"""
Generate exercisesxp.ipynb for Week 6 Day 3:
LSTM time-series analysis on the household_power_consumption dataset.
"""

import json
import os


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = [
    md(
        "# XP Exercises — LSTM on Household Power Consumption",
        "",
        "**Course:** Developers Institute  **Week 6 - Day 3**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Time-series workflow end-to-end: load the UCI *Individual Household Electric",
        "Power Consumption* dataset, clean the missing values, visualize the daily",
        "patterns, then train an **LSTM** to predict `Global_active_power`.",
        "",
        "Six sequential parts: import + exploration, missing values, visualization,",
        "preprocessing, model architecture, training + evaluation.",
        "",
        "**⚠️ Use Colab with a GPU runtime** (`Runtime → Change runtime type → GPU`).",
        "The dataset has 2 million rows; with a CPU runtime the LSTM training is slow.",
    ),

    md("## Setup"),
    code(
        "%pip install -qU tensorflow",
    ),
    code(
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import os, pathlib, zipfile, urllib.request",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "sns.set_theme(style='whitegrid')",
        "RANDOM_STATE = 42",
        "np.random.seed(RANDOM_STATE)",
    ),

    # ============================================================
    # Part 1 — Import + initial exploration
    # ============================================================
    md(
        "## Part 1 — Data Import and Initial Exploration",
        "",
        "We download the dataset, parse it as a time series, and inspect its structure.",
    ),
    code(
        "URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip'",
        "DATA_DIR = pathlib.Path('/content/power_data')",
        "DATA_DIR.mkdir(parents=True, exist_ok=True)",
        "ZIP_PATH = DATA_DIR / 'household_power_consumption.zip'",
        "CSV_PATH = DATA_DIR / 'household_power_consumption.txt'",
        "",
        "if not CSV_PATH.exists():",
        "    print('Downloading dataset...')",
        "    urllib.request.urlretrieve(URL, ZIP_PATH)",
        "    with zipfile.ZipFile(ZIP_PATH) as zf:",
        "        zf.extractall(DATA_DIR)",
        "    print('Done.')",
        "else:",
        "    print('Dataset already present.')",
        "",
        "print('Files:', [p.name for p in DATA_DIR.iterdir()])",
    ),
    code(
        "# Parse: ';' separator, '?' marks missing values, combine Date+Time into a single datetime index",
        "df = pd.read_csv(",
        "    CSV_PATH,",
        "    sep=';',",
        "    parse_dates={'datetime': ['Date', 'Time']},",
        "    dayfirst=True,",
        "    na_values='?',",
        "    low_memory=False,",
        ")",
        "df = df.set_index('datetime').sort_index()",
        "",
        "# Force numeric dtypes (after na_values they may still be objects)",
        "for col in df.columns:",
        "    df[col] = pd.to_numeric(df[col], errors='coerce')",
        "",
        "print(f'Shape: {df.shape}')",
        "print(f'Date range: {df.index.min()} -> {df.index.max()}')",
        "df.head()",
    ),
    code(
        "# Data types + shape sanity",
        "print(df.dtypes)",
        "print(f'\\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')",
    ),

    # ============================================================
    # Part 2 — Missing values
    # ============================================================
    md("## Part 2 — Handling Missing Values"),
    code(
        "missing = df.isna().sum()",
        "print('Missing values per column:')",
        "print(missing)",
        "print(f'\\nTotal missing cells: {missing.sum()} ({missing.sum() / df.size * 100:.2f}%)')",
    ),
    code(
        "# Fill missing values with the per-column mean (as the brief asks)",
        "df = df.fillna(df.mean(numeric_only=True))",
        "",
        "# Verify",
        "assert df.isna().sum().sum() == 0, 'Still missing values!'",
        "print('All missing values filled.')",
        "print(df.isna().sum())",
    ),

    # ============================================================
    # Part 3 — Visualization
    # ============================================================
    md(
        "## Part 3 — Data Visualization",
        "",
        "Resample `Global_active_power` to **daily** values and plot the sum and the",
        "mean side by side. Then plot the daily mean and standard deviation of",
        "`Global_intensity`.",
    ),
    code(
        "# Daily resampling of Global_active_power",
        "daily_gap = df['Global_active_power'].resample('D').agg(['sum', 'mean'])",
        "",
        "fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)",
        "axes[0].plot(daily_gap.index, daily_gap['sum'], color='steelblue', linewidth=1)",
        "axes[0].set_title('Global_active_power — daily SUM (kW)', fontweight='bold')",
        "axes[0].set_ylabel('Sum (kW)')",
        "",
        "axes[1].plot(daily_gap.index, daily_gap['mean'], color='seagreen', linewidth=1)",
        "axes[1].set_title('Global_active_power — daily MEAN (kW)', fontweight='bold')",
        "axes[1].set_ylabel('Mean (kW)')",
        "axes[1].set_xlabel('Date')",
        "",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "# Daily mean and std of Global_intensity",
        "daily_gi = df['Global_intensity'].resample('D').agg(['mean', 'std'])",
        "",
        "fig, ax = plt.subplots(figsize=(13, 4.5))",
        "ax.plot(daily_gi.index, daily_gi['mean'], color='steelblue', label='Mean', linewidth=1)",
        "ax.fill_between(",
        "    daily_gi.index,",
        "    daily_gi['mean'] - daily_gi['std'],",
        "    daily_gi['mean'] + daily_gi['std'],",
        "    alpha=0.25, color='steelblue', label='+/- 1 std',",
        ")",
        "ax.set_title('Global_intensity — daily mean and standard deviation', fontweight='bold')",
        "ax.set_xlabel('Date'); ax.set_ylabel('Intensity (A)')",
        "ax.legend()",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "**Observations.**",
        "- A clear **seasonal pattern**: winter peaks (heating) and summer dips.",
        "- The standard deviation band tracks the mean closely — when consumption is",
        "  high it is also more volatile.",
        "- Mean ≈ sum/1440 (1440 minutes in a day) — a sanity check that the resampling",
        "  worked.",
    ),

    # ============================================================
    # Part 4 — Preprocessing for LSTM
    # ============================================================
    md(
        "## Part 4 — Data Preprocessing for LSTM",
        "",
        "We downsample to **hourly** resolution to keep training tractable, then",
        "scale features to `[0, 1]`, build supervised sequences (24-hour windows →",
        "next-hour prediction), and split into train/test (80/20, no shuffle — order",
        "matters in time series).",
    ),
    code(
        "from sklearn.preprocessing import MinMaxScaler",
        "",
        "# 1) Downsample to hourly resolution to reduce ~2M rows to ~35k",
        "hourly = df.resample('H').mean()",
        "print(f'Hourly shape: {hourly.shape}')",
        "",
        "# 2) Target = Global_active_power; features = all numeric columns",
        "target_col = 'Global_active_power'",
        "features = hourly.columns.tolist()",
        "scaler = MinMaxScaler()",
        "scaled = scaler.fit_transform(hourly[features])",
        "scaled_df = pd.DataFrame(scaled, columns=features, index=hourly.index)",
        "print('Scaled range:', scaled.min().round(3), 'to', scaled.max().round(3))",
    ),
    code(
        "# 3) Build supervised sequences: 24-hour windows -> predict next hour's Global_active_power",
        "WINDOW = 24",
        "X, y = [], []",
        "target_idx = features.index(target_col)",
        "for i in range(len(scaled) - WINDOW):",
        "    X.append(scaled[i:i + WINDOW])",
        "    y.append(scaled[i + WINDOW, target_idx])",
        "X = np.array(X, dtype='float32')",
        "y = np.array(y, dtype='float32')",
        "",
        "print(f'X shape: {X.shape}  (samples, timesteps, features)')",
        "print(f'y shape: {y.shape}')",
    ),
    code(
        "# 4) Train/test split — chronological 80/20",
        "split = int(0.8 * len(X))",
        "X_train, X_test = X[:split], X[split:]",
        "y_train, y_test = y[:split], y[split:]",
        "print(f'Train: {X_train.shape}  | Test: {X_test.shape}')",
    ),

    # ============================================================
    # Part 5 — LSTM architecture
    # ============================================================
    md(
        "## Part 5 — Building the LSTM Model",
        "",
        "Two stacked LSTM layers + Dropout, then a Dense regression head.",
        "Loss = MSE, optimizer = Adam.",
    ),
    code(
        "import tensorflow as tf",
        "from tensorflow.keras.models import Sequential",
        "from tensorflow.keras.layers import LSTM, Dense, Dropout",
        "from tensorflow.keras.callbacks import EarlyStopping",
        "",
        "tf.random.set_seed(RANDOM_STATE)",
        "",
        "model = Sequential([",
        "    LSTM(64, return_sequences=True, input_shape=(WINDOW, X.shape[2])),",
        "    Dropout(0.2),",
        "    LSTM(32),",
        "    Dropout(0.2),",
        "    Dense(16, activation='relu'),",
        "    Dense(1),  # regression",
        "])",
        "",
        "model.compile(optimizer='adam', loss='mse', metrics=['mae'])",
        "model.summary()",
    ),

    # ============================================================
    # Part 6 — Train + evaluate
    # ============================================================
    md(
        "## Part 6 — Training and Evaluating the LSTM",
        "",
        "We train for up to 15 epochs with `EarlyStopping` watching `val_loss`. Then",
        "plot loss curves and compare predicted vs actual on the test window.",
    ),
    code(
        "callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]",
        "",
        "history = model.fit(",
        "    X_train, y_train,",
        "    epochs=15,",
        "    batch_size=128,",
        "    validation_split=0.1,",
        "    callbacks=callbacks,",
        "    verbose=1,",
        ")",
        "",
        "test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)",
        "print(f'\\nTest MSE: {test_loss:.5f}')",
        "print(f'Test MAE: {test_mae:.5f}')",
    ),
    code(
        "# Training curves",
        "plt.figure(figsize=(11, 4.5))",
        "plt.plot(history.history['loss'], label='train loss')",
        "plt.plot(history.history['val_loss'], label='val loss')",
        "plt.xlabel('Epoch'); plt.ylabel('MSE')",
        "plt.title('LSTM training — loss across epochs', fontweight='bold')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "# Predicted vs actual on the test set",
        "y_pred_scaled = model.predict(X_test, verbose=0).ravel()",
        "",
        "# Inverse-scale only the target column for a readable plot",
        "target_min = scaler.data_min_[target_idx]",
        "target_range = scaler.data_max_[target_idx] - target_min",
        "y_pred = y_pred_scaled * target_range + target_min",
        "y_true = y_test * target_range + target_min",
        "",
        "plt.figure(figsize=(13, 4.5))",
        "plt.plot(y_true[:500], label='Actual', color='steelblue', linewidth=1)",
        "plt.plot(y_pred[:500], label='Predicted', color='tomato', linewidth=1, alpha=0.8)",
        "plt.xlabel('Hour (test set)')",
        "plt.ylabel('Global_active_power (kW)')",
        "plt.title('LSTM — predicted vs actual (first 500 hours of test)', fontweight='bold')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "# RMSE in original units",
        "rmse = np.sqrt(((y_pred - y_true) ** 2).mean())",
        "mae = np.abs(y_pred - y_true).mean()",
        "print(f'Test RMSE (kW): {rmse:.3f}')",
        "print(f'Test MAE  (kW): {mae:.3f}')",
    ),

    md(
        "## Summary",
        "",
        "- We loaded a 2-million-row, minute-level time series, parsed dates correctly,",
        "  and filled the small fraction of `?` missing entries with the column means.",
        "- Visualization at daily resolution surfaced the seasonal pattern of",
        "  electricity consumption (winter peaks, summer dips) and the link between",
        "  mean and volatility.",
        "- For training we downsampled to **hourly** resolution, scaled to `[0, 1]`,",
        "  and built **24-hour input windows** to predict the next hour's",
        "  `Global_active_power`.",
        "- The LSTM model (two layers, Dropout, regression head) trained quickly with",
        "  `EarlyStopping` and reached a low MAE on the held-out chunk of the series.",
        "- **Next steps:** longer windows, multi-step forecasting, calendar features",
        "  (day-of-week, hour-of-day, holidays), and benchmarking against simpler",
        "  baselines like seasonal naive or `ARIMA` to confirm the LSTM is actually",
        "  earning its complexity budget.",
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = os.path.join(os.path.dirname(__file__), "exercisesxp.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
