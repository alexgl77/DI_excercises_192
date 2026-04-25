"""
Daily Challenge - Comprehensive Mobile Price Analysis

Dataset: Mobile Price Classification (Kaggle / Abhishek Sharma).
Target: price_range (0=low, 1=medium, 2=high, 3=very high).

Estructura:
1. Data Loading and Exploration
2. Data Cleaning and Preprocessing
3. Statistical Analysis (NumPy + SciPy)
4. Hypothesis Testing
5. Visualizaciones (Matplotlib)
6. Insight Synthesis and Conclusion
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "mobile_train.csv")
DATA_URL = ("https://raw.githubusercontent.com/Nithya-Vasudevan/"
            "Mobile-Price-Classification/master/train.csv")


def ensure_dataset() -> str:
    if not os.path.exists(CSV_PATH):
        print(f"Descargando dataset desde {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, CSV_PATH)
    return CSV_PATH


# ----------------------------------------------------------------------
# 1. Data Loading and Exploration
# ----------------------------------------------------------------------
print("=" * 70)
print("1. DATA LOADING AND EXPLORATION")
print("=" * 70)

ensure_dataset()
df = pd.read_csv(CSV_PATH)

print(f"Shape: {df.shape}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nDtypes:")
print(df.dtypes)
print(f"\nDistribucion de price_range (target):")
print(df["price_range"].value_counts().sort_index())
print(f"\nDescribe:")
print(df.describe().round(3))


# ----------------------------------------------------------------------
# 2. Data Cleaning and Preprocessing
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. DATA CLEANING AND PREPROCESSING")
print("=" * 70)

print(f"Valores nulos por columna (suma): {df.isna().sum().sum()}")
print(f"Filas duplicadas: {df.duplicated().sum()}")

# Identificar features binarias y continuas
binary_cols = [c for c in df.columns if set(df[c].unique()).issubset({0, 1})]
continuous_cols = [c for c in df.columns
                   if c not in binary_cols and c != "price_range"]
print(f"\nFeatures binarias ({len(binary_cols)}): {binary_cols}")
print(f"Features continuas ({len(continuous_cols)}): {continuous_cols}")

# Anomalias: sc_w=0 y px_height=0 son fisicamente imposibles. Las marcamos
# como NaN y las imputamos con la mediana del grupo.
suspicious = {"sc_w": 0, "px_height": 0}
for col, bad_val in suspicious.items():
    n_bad = (df[col] == bad_val).sum()
    if n_bad > 0:
        print(f"  {col}: {n_bad} valores == {bad_val} sospechosos -> imputo "
              f"con mediana del price_range")
        df.loc[df[col] == bad_val, col] = np.nan
        df[col] = df.groupby("price_range")[col].transform(
            lambda s: s.fillna(s.median()))

# Todo el dataset es ya numerico; las binarias funcionan como dummies.
# Si hubiese categoricas tipo string, aqui aplicariamos pd.get_dummies.

print(f"\nValores nulos tras imputacion: {df.isna().sum().sum()}")


# ----------------------------------------------------------------------
# 3. Statistical Analysis con NumPy y SciPy
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. STATISTICAL ANALYSIS")
print("=" * 70)

print("\nResumen estadistico features continuas (NumPy + SciPy):")
rows = []
for col in continuous_cols:
    arr = df[col].to_numpy()
    rows.append({
        "feature": col,
        "mean": np.mean(arr),
        "median": np.median(arr),
        "mode": stats.mode(arr, keepdims=False).mode,
        "range": np.ptp(arr),
        "var": np.var(arr, ddof=1),
        "std": np.std(arr, ddof=1),
        "skew": stats.skew(arr),
        "kurt": stats.kurtosis(arr),
    })
summary = pd.DataFrame(rows).set_index("feature").round(3)
print(summary)

# Correlacion (Pearson) feature -> target via SciPy
print("\nCorrelaciones feature <-> price_range (Pearson, SciPy):")
target = df["price_range"].to_numpy()
corr_rows = []
for col in df.columns:
    if col == "price_range":
        continue
    r, p = stats.pearsonr(df[col].to_numpy(), target)
    corr_rows.append({"feature": col, "pearson_r": r, "p_value": p})
corr_table = (pd.DataFrame(corr_rows)
              .sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
              .reset_index(drop=True))
print(corr_table.round(4))

# Spearman para capturar relaciones monotonicas no lineales
spearman_r, spearman_p = stats.spearmanr(df["ram"], df["price_range"])
print(f"\nSpearman(ram, price_range): rho={spearman_r:.4f}, p={spearman_p:.4e}")


# ----------------------------------------------------------------------
# 4. Hypothesis Testing
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. HYPOTHESIS TESTING")
print("=" * 70)

# 4a. ANOVA: hay diferencias en RAM entre los 4 price_range?
groups_ram = [df.loc[df["price_range"] == k, "ram"].to_numpy() for k in range(4)]
f_stat, anova_p = stats.f_oneway(*groups_ram)
print(f"ANOVA (RAM por price_range): F={f_stat:.2f}, p={anova_p:.4e}")
if anova_p < 0.05:
    print("  => Diferencias significativas en RAM entre los 4 grupos.")

# 4b. T-test independiente: RAM en 'low cost' vs 'very high cost'
ram_low = df.loc[df["price_range"] == 0, "ram"].to_numpy()
ram_high = df.loc[df["price_range"] == 3, "ram"].to_numpy()
t_stat, t_p = stats.ttest_ind(ram_low, ram_high, equal_var=False)
print(f"\nT-test RAM low (0) vs very-high (3): t={t_stat:.2f}, p={t_p:.4e}")
print(f"  Media RAM low : {ram_low.mean():.0f} MB")
print(f"  Media RAM high: {ram_high.mean():.0f} MB")

# 4c. Chi-cuadrado: 4G y price_range estan asociados?
contingency = pd.crosstab(df["four_g"], df["price_range"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi2 (four_g vs price_range): chi2={chi2:.3f}, dof={dof}, p={chi_p:.4f}")
if chi_p < 0.05:
    print("  => Asociacion significativa entre 4G y precio.")
else:
    print("  => No hay evidencia de asociacion entre 4G y precio.")

# 4d. Mann-Whitney U: peso (mobile_wt) entre celulares baratos vs caros
wt_low = df.loc[df["price_range"] == 0, "mobile_wt"].to_numpy()
wt_high = df.loc[df["price_range"] == 3, "mobile_wt"].to_numpy()
u_stat, u_p = stats.mannwhitneyu(wt_low, wt_high, alternative="two-sided")
print(f"\nMann-Whitney U (mobile_wt low vs high): U={u_stat:.0f}, p={u_p:.4e}")


# ----------------------------------------------------------------------
# 5. Visualizaciones
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. VISUALIZACIONES")
print("=" * 70)

# 5a. Histogramas de las 6 features mas correlacionadas con el target
top_features = corr_table["feature"].head(6).tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, feat in zip(axes.flat, top_features):
    ax.hist(df[feat], bins=30, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.set_title(f"{feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Frecuencia")
    ax.grid(True, linestyle="--", alpha=0.4)
fig.suptitle("Histogramas - top 6 features por correlacion con price_range")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_dc_histograms.png"), dpi=120)
plt.close()

# 5b. Boxplots de RAM por price_range (la feature mas predictiva)
fig, ax = plt.subplots(figsize=(8, 5))
data_box = [df.loc[df["price_range"] == k, "ram"] for k in range(4)]
ax.boxplot(data_box, labels=["0 (low)", "1", "2", "3 (very high)"])
ax.set_title("RAM por price_range")
ax.set_xlabel("price_range")
ax.set_ylabel("RAM (MB)")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_dc_ram_boxplot.png"), dpi=120)
plt.close()

# 5c. Scatter RAM vs Battery, color por price_range
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df["battery_power"], df["ram"],
                     c=df["price_range"], cmap="viridis",
                     alpha=0.6, s=20)
ax.set_xlabel("battery_power (mAh)")
ax.set_ylabel("ram (MB)")
ax.set_title("Scatter battery_power vs ram (color = price_range)")
plt.colorbar(scatter, label="price_range")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_dc_scatter_ram_battery.png"), dpi=120)
plt.close()

# 5d. Heatmap de correlaciones (matriz completa)
corr_full = df.corr().to_numpy()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_full, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(df.columns)))
ax.set_yticklabels(df.columns, fontsize=8)
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        v = corr_full[i, j]
        if abs(v) > 0.15:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=7)
fig.colorbar(im, ax=ax)
ax.set_title("Heatmap de correlaciones")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_dc_heatmap.png"), dpi=120)
plt.close()

# 5e. Bar chart de correlaciones absolutas con el target
fig, ax = plt.subplots(figsize=(10, 6))
bar_data = corr_table.copy()
bar_data["abs_r"] = bar_data["pearson_r"].abs()
bar_data = bar_data.sort_values("abs_r", ascending=True)
colors = ["#d62728" if r < 0 else "#2ca02c" for r in bar_data["pearson_r"]]
ax.barh(bar_data["feature"], bar_data["abs_r"], color=colors)
ax.set_xlabel("|correlacion| con price_range")
ax.set_title("Importancia univariada (Pearson) de cada feature")
ax.grid(True, axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mp_dc_feature_importance.png"), dpi=120)
plt.close()

print("Graficos guardados:")
for f in ["mp_dc_histograms.png", "mp_dc_ram_boxplot.png",
          "mp_dc_scatter_ram_battery.png", "mp_dc_heatmap.png",
          "mp_dc_feature_importance.png"]:
    print(f"  {f}")


# ----------------------------------------------------------------------
# 6. Insight Synthesis and Conclusion
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. SUMMARY AND INSIGHTS")
print("=" * 70)

top3 = corr_table.head(3)
print(f"""
Hallazgos principales:

- Dataset balanceado: 500 ejemplos por price_range. No hay sesgo de clase.

- Top 3 features por correlacion con el target:
  1. {top3.iloc[0]['feature']:15s} r = {top3.iloc[0]['pearson_r']:.3f}
  2. {top3.iloc[1]['feature']:15s} r = {top3.iloc[1]['pearson_r']:.3f}
  3. {top3.iloc[2]['feature']:15s} r = {top3.iloc[2]['pearson_r']:.3f}
  -> RAM es por lejos el mejor predictor lineal del rango de precio.

- ANOVA confirma diferencias significativas en RAM entre los 4 rangos
  (F={f_stat:.2f}, p<<0.001). El t-test low vs very-high da p={t_p:.2e},
  mostrando que la RAM crece de forma marcada con el precio
  ({ram_low.mean():.0f} MB en low vs {ram_high.mean():.0f} MB en very-high).

- Chi2 four_g vs price_range: p={chi_p:.3f}. {"Hay" if chi_p < 0.05 else "NO hay"}
  asociacion significativa entre tener 4G y la categoria de precio.

- Caracteristicas que en intuicion deberian importar (camara, peso,
  resolucion, talk_time) tienen correlacion debil/nula. Esto sugiere que
  los fabricantes no escalan estos atributos linealmente con el precio.

- Variables binarias (blue, dual_sim, four_g, three_g, touch_screen, wifi)
  presentan correlacion practicamente nula. La diferenciacion de precio en
  este dataset se da por hardware "duro" (RAM, CPU, bateria, pantalla),
  no por features de conectividad.

- Skewness y kurtosis cercanas a cero en la mayoria de features continuas
  indica distribuciones aproximadamente uniformes (consistente con
  generacion sintetica del dataset original).

Conclusion: para clasificar un celular en su rango de precio, el modelo
deberia priorizar RAM, pixeles totales (px_width * px_height), bateria y
peso. Las binarias aportan poco por si solas.
""")
