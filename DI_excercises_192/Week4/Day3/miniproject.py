"""
Mini-project: Advanced Statistical Analysis of Apple Inc. Stock Data (AAPL, 1981-2023)

Estructura:
1. Data Loading and Exploration
2. Data Visualization (precios, volumen, candlestick)
3. Statistical Analysis (resumen + media movil)
4. Hypothesis Testing (t-test entre anios + normalidad de retornos)
5. Advanced Statistical Techniques (convolucion, correlacion entre MA y volumen)
6. Summary and Insights
7. Reflection

Si yfinance esta instalado y aapl.csv no existe, lo descarga automaticamente.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "aapl.csv")


def ensure_dataset() -> str:
    if os.path.exists(CSV_PATH):
        return CSV_PATH
    print("Descargando AAPL via yfinance ...")
    import yfinance as yf
    df = yf.download("AAPL", start="1981-01-01", end="2024-01-01",
                     progress=False, auto_adjust=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.to_csv(CSV_PATH)
    return CSV_PATH


# ----------------------------------------------------------------------
# 1. Data Loading and Exploration
# ----------------------------------------------------------------------
print("=" * 70)
print("1. DATA LOADING AND EXPLORATION")
print("=" * 70)

ensure_dataset()
df = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")
df = df.sort_index()

print(f"Shape: {df.shape}")
print(f"Rango de fechas: {df.index.min().date()} -> {df.index.max().date()}")
print("\nDtypes:")
print(df.dtypes)
print("\nNulos por columna:")
print(df.isna().sum())
print("\nPrimeras filas:")
print(df.head(3).round(4))

# Frecuencia: dias habiles (Bolsa NY); chequeamos gaps
all_business_days = pd.bdate_range(df.index.min(), df.index.max())
gaps = len(all_business_days) - len(df)
print(f"\nDias habiles teoricos: {len(all_business_days)}, observados: {len(df)}, "
      f"gaps (feriados/etc.): {gaps}")


# ----------------------------------------------------------------------
# 2. Data Visualization
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. DATA VISUALIZATION")
print("=" * 70)

# 2a. Precio de cierre + volumen en figura compartida
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1)
ax1.set_title("AAPL - precio de cierre 1981-2023")
ax1.set_ylabel("Precio (USD)")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.fill_between(df.index, df["Volume"] / 1e6, color="#ff7f0e", alpha=0.7)
ax2.set_ylabel("Volumen (M acciones)")
ax2.set_xlabel("Fecha")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_close_volume.png"), dpi=120)
plt.close()

# 2b. Candlestick del ultimo anio (para que sea legible)
last_year = df.loc[df.index >= df.index.max() - pd.Timedelta(days=365)].copy()

fig, ax = plt.subplots(figsize=(12, 5))
width = 0.6
for i, (date, row) in enumerate(last_year.iterrows()):
    color = "#2ca02c" if row["Close"] >= row["Open"] else "#d62728"
    # mecha (high-low)
    ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1)
    # cuerpo (open-close)
    body_low = min(row["Open"], row["Close"])
    body_height = abs(row["Close"] - row["Open"])
    ax.add_patch(Rectangle((i - width / 2, body_low), width,
                           body_height if body_height > 0 else 0.01,
                           facecolor=color, edgecolor=color))

# eje x con fechas distribuidas
n_ticks = 8
tick_idx = np.linspace(0, len(last_year) - 1, n_ticks).astype(int)
ax.set_xticks(tick_idx)
ax.set_xticklabels([last_year.index[i].strftime("%Y-%m-%d") for i in tick_idx],
                   rotation=30, ha="right")
ax.set_title(f"Candlestick AAPL - ultimos 12 meses ({last_year.index.min().date()} -> {last_year.index.max().date()})")
ax.set_ylabel("Precio (USD)")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_candlestick.png"), dpi=120)
plt.close()

print("Graficos guardados: mp_close_volume.png, mp_candlestick.png")


# ----------------------------------------------------------------------
# 3. Statistical Analysis
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. STATISTICAL ANALYSIS")
print("=" * 70)

stats_summary = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].agg(
    ["mean", "median", "std", "min", "max"]
).round(4)
print("Resumen estadistico:")
print(stats_summary)

# Medias moviles 50 y 200 dias del cierre
df["MA50"] = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df["Close"], color="#1f77b4", alpha=0.5, label="Close")
ax.plot(df.index, df["MA50"], color="#ff7f0e", label="MA 50")
ax.plot(df.index, df["MA200"], color="#d62728", label="MA 200")
ax.set_title("AAPL - cierre con medias moviles 50 y 200 dias")
ax.set_ylabel("Precio (USD)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_moving_averages.png"), dpi=120)
plt.close()
print("Grafico guardado: mp_moving_averages.png")


# ----------------------------------------------------------------------
# 4. Hypothesis Testing
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. HYPOTHESIS TESTING")
print("=" * 70)

# 4a. T-test entre los cierres de 2020 y 2022
df["Year"] = df.index.year
year_a, year_b = 2020, 2022
close_a = df.loc[df["Year"] == year_a, "Close"].to_numpy()
close_b = df.loc[df["Year"] == year_b, "Close"].to_numpy()

t_stat, p_value = stats.ttest_ind(close_a, close_b, equal_var=False)
print(f"H0: media(Close {year_a}) == media(Close {year_b})")
print(f"  {year_a}: n={len(close_a)}, media={close_a.mean():.2f}, sd={close_a.std(ddof=1):.2f}")
print(f"  {year_b}: n={len(close_b)}, media={close_b.mean():.2f}, sd={close_b.std(ddof=1):.2f}")
print(f"  t = {t_stat:.4f}, p-value = {p_value:.4e}")
if p_value < 0.05:
    print(f"  => Rechazamos H0: la media difiere significativamente entre {year_a} y {year_b}.")
else:
    print(f"  => No rechazamos H0.")

# 4b. Distribucion de retornos diarios + test de normalidad
df["Return"] = df["Close"].pct_change()
returns = df["Return"].dropna().to_numpy()

print(f"\nRetornos diarios: n={len(returns)}, media={returns.mean():.5f}, "
      f"sd={returns.std(ddof=1):.5f}, skew={stats.skew(returns):.3f}, "
      f"kurtosis={stats.kurtosis(returns):.3f}")

# Test de normalidad: Jarque-Bera (apto para muestras grandes; Shapiro queda
# limitado por debajo de ~5000 datos)
jb_stat, jb_p = stats.jarque_bera(returns)
print(f"Jarque-Bera: estadistico={jb_stat:.2f}, p-value={jb_p:.4e}")
if jb_p < 0.05:
    print("  => Rechazamos normalidad: los retornos diarios NO son normales "
          "(colas pesadas, exceso de curtosis).")
else:
    print("  => No se rechaza normalidad.")

# Histograma + qq-plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(returns, bins=80, color="#1f77b4", edgecolor="black", alpha=0.7,
         density=True)
# overlay normal teorica
x = np.linspace(returns.min(), returns.max(), 300)
ax1.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
         color="red", linewidth=2, label="Normal teorica")
ax1.set_title("Distribucion de retornos diarios")
ax1.set_xlabel("Retorno")
ax1.set_ylabel("Densidad")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

stats.probplot(returns, dist="norm", plot=ax2)
ax2.set_title("Q-Q plot vs Normal")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_returns_distribution.png"), dpi=120)
plt.close()
print("Grafico guardado: mp_returns_distribution.png")


# ----------------------------------------------------------------------
# 5. Advanced Statistical Techniques (Bonus)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. ADVANCED TECHNIQUES")
print("=" * 70)

# 5a. Media movil via np.convolve (alternativa manual al rolling de pandas)
window = 30
kernel = np.ones(window) / window
close_arr = df["Close"].to_numpy()
ma_conv = np.convolve(close_arr, kernel, mode="valid")
ma_dates = df.index[window - 1:]

print(f"Media movil 30 dias via np.convolve: shape={ma_conv.shape}, "
      f"primeros 3 valores={np.round(ma_conv[:3], 4)}")

# Comparamos con rolling de pandas (deben ser identicos)
ma_pd = df["Close"].rolling(window=window).mean().dropna().to_numpy()
print(f"Maxima diferencia rolling vs convolve: {np.abs(ma_conv - ma_pd).max():.2e}")

# 5b. Correlacion entre medias moviles del cierre y del volumen
df["MA30_Close"] = df["Close"].rolling(30).mean()
df["MA30_Volume"] = df["Volume"].rolling(30).mean()

corr_df = df[["MA30_Close", "MA30_Volume", "Close", "Volume"]].dropna()
corr_matrix = np.corrcoef(corr_df.to_numpy(), rowvar=False)
corr_pd = pd.DataFrame(corr_matrix, index=corr_df.columns, columns=corr_df.columns)
print("\nMatriz de correlacion (np.corrcoef):")
print(corr_pd.round(3))

# Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_pd.columns)))
ax.set_xticklabels(corr_pd.columns, rotation=30, ha="right")
ax.set_yticks(range(len(corr_pd.columns)))
ax.set_yticklabels(corr_pd.columns)
for i in range(len(corr_pd.columns)):
    for j in range(len(corr_pd.columns)):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax)
ax.set_title("Correlaciones - precio, volumen, MA30")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_correlation.png"), dpi=120)
plt.close()
print("Grafico guardado: mp_correlation.png")

# 5c. Correlacion movil entre MA del precio y del volumen (60 dias)
roll_corr = (df["MA30_Close"]
             .rolling(60)
             .corr(df["MA30_Volume"]))
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(roll_corr.index, roll_corr, color="#9467bd")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Correlacion movil 60d entre MA30(Close) y MA30(Volume)")
ax.set_ylabel("Correlacion")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_rolling_correlation.png"), dpi=120)
plt.close()
print("Grafico guardado: mp_rolling_correlation.png")


# ----------------------------------------------------------------------
# 6. Summary and Insights
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. SUMMARY AND INSIGHTS")
print("=" * 70)

total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
years_span = (df.index.max() - df.index.min()).days / 365.25
cagr = (1 + total_return) ** (1 / years_span) - 1

print(f"""
- Periodo cubierto: {df.index.min().date()} -> {df.index.max().date()} ({years_span:.1f} anios).
- Precio inicial: {df['Close'].iloc[0]:.4f} USD ; final: {df['Close'].iloc[-1]:.2f} USD.
- Retorno total acumulado: {total_return * 100:,.0f}%  (CAGR ~ {cagr * 100:.2f}%).
- Volumen medio diario: {df['Volume'].mean() / 1e6:.1f} millones de acciones.
- T-test cierre {year_a} vs {year_b}: t={t_stat:.2f}, p={p_value:.2e}
  -> los precios medios entre esos dos anios son estadisticamente distintos.
- Retornos diarios: media practicamente cero ({returns.mean():.5f}), sd={returns.std(ddof=1):.4f},
  curtosis={stats.kurtosis(returns):.2f} (>>0 -> colas pesadas).
- Jarque-Bera rechaza normalidad (p<<0.05): el modelo Gaussiano subestima
  el riesgo de movimientos extremos. Hecho conocido en finanzas.
- Correlacion contemporanea entre Close y Volume: {corr_pd.loc['Close', 'Volume']:.2f}
  -> debil. La relacion es mas estable cuando se suaviza con MA30.
- np.convolve replica exactamente el rolling.mean de pandas, lo que valida
  el equivalente como filtro lineal de promedio.
""")


# ----------------------------------------------------------------------
# 7. Reflection
# ----------------------------------------------------------------------
print("=" * 70)
print("7. REFLECTION")
print("=" * 70)
print("""
Desafios encontrados y soluciones:

- Datos: yfinance descarga con encabezado multi-index (Price/Ticker). Se
  resolvio aplanando columnas a una sola fila al guardar el CSV.

- Frecuencia: la serie es diaria habil, no calendario. Se uso pd.bdate_range
  para diagnosticar gaps (feriados de mercado) en lugar de asumir frecuencia
  D continua.

- Tamanio para tests de normalidad: con ~10.840 retornos, el test de Shapiro
  es poco confiable. Se uso Jarque-Bera, que aprovecha skewness y kurtosis y
  es estandar en analisis financiero.

- Visualizacion de candlestick: en vez de depender de mplfinance (no instalado),
  se dibujaron rectangulos + lineas con matplotlib.patches. Se grafico solo
  el ultimo anio para que las velas sean legibles.

- Convolucion vs rolling: ambos producen el mismo resultado para una media
  movil simple, pero np.convolve permite kernels arbitrarios (exponenciales,
  ponderados) y se integra naturalmente con tecnicas de signal processing.
""")
