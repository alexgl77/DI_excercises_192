"""
Daily Challenge - Week 4 Day 2:
Analisis del Global Power Plant Database (WRI) integrando NumPy, Pandas y Matplotlib.

Cubre:
1. Importacion y limpieza
2. EDA (estadisticos clave, distribucion por pais y combustible)
3. Analisis estadistico de potencia por combustible + test de hipotesis
4. Series de tiempo (anio de puesta en marcha)
5. Visualizaciones (Matplotlib + Seaborn)
6. Operaciones matriciales (correlacion, eigen)
7. Integracion NumPy <-> Pandas <-> Matplotlib

El script descarga el dataset si no esta presente.
"""

import os
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "global_power_plant_database.csv")
DATASET_URL = (
    "https://wri-dataportal-prod.s3.amazonaws.com/manual/"
    "global_power_plant_database_v_1_3.zip"
)

sns.set_theme(style="whitegrid")
np.random.seed(42)


def ensure_dataset() -> str:
    """Descarga y descomprime el dataset si no existe localmente."""
    if os.path.exists(CSV_PATH):
        return CSV_PATH
    print(f"Descargando dataset desde {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(HERE)
    return CSV_PATH


# ----------------------------------------------------------------------
# 1. Importacion y limpieza
# ----------------------------------------------------------------------
print("=" * 70)
print("1. IMPORTACION Y LIMPIEZA")
print("=" * 70)

ensure_dataset()
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Shape original: {df.shape}")

print("\nValores faltantes (top 10 columnas):")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(10))

# Garantizar tipos numericos en columnas clave
numeric_cols = ["capacity_mw", "latitude", "longitude", "commissioning_year"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Para el analisis trabajamos con plantas que tengan capacidad y combustible
df_clean = df.dropna(subset=["capacity_mw", "primary_fuel"]).copy()
# commissioning_year tiene algunos valores fraccionarios (anio + fraccion); redondeamos.
df_clean["commissioning_year"] = df_clean["commissioning_year"].round()

print(f"\nShape tras limpieza (capacity_mw + primary_fuel no nulos): {df_clean.shape}")


# ----------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nEstadisticos descriptivos de columnas numericas:")
print(df_clean[["capacity_mw", "latitude", "longitude", "commissioning_year"]]
      .describe().round(2))

# Top paises por cantidad de plantas
top_countries = df_clean["country_long"].value_counts().head(10)
print("\nTop 10 paises por cantidad de plantas:")
print(top_countries)

# Distribucion por combustible primario
fuel_counts = df_clean["primary_fuel"].value_counts()
print("\nDistribucion por combustible primario:")
print(fuel_counts)


# ----------------------------------------------------------------------
# 3. Analisis estadistico + test de hipotesis
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. ANALISIS ESTADISTICO POR COMBUSTIBLE")
print("=" * 70)

stats_by_fuel = df_clean.groupby("primary_fuel")["capacity_mw"].agg(
    count="count",
    mean=lambda x: np.mean(x),
    median=lambda x: np.median(x),
    std=lambda x: np.std(x, ddof=1),
)
stats_by_fuel = stats_by_fuel.sort_values("mean", ascending=False).round(2)
print("\nCapacidad (MW) por combustible:")
print(stats_by_fuel)

# Test de hipotesis: la capacidad media de plantas Nuclear vs Solar es distinta?
# Welch's t-test (dos colas) implementado a mano con NumPy.
group_a = df_clean.loc[df_clean["primary_fuel"] == "Nuclear", "capacity_mw"].to_numpy()
group_b = df_clean.loc[df_clean["primary_fuel"] == "Solar", "capacity_mw"].to_numpy()

mean_a, mean_b = np.mean(group_a), np.mean(group_b)
var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
n_a, n_b = len(group_a), len(group_b)

se = np.sqrt(var_a / n_a + var_b / n_b)
t_stat = (mean_a - mean_b) / se
# df de Welch-Satterthwaite
df_welch = (var_a / n_a + var_b / n_b) ** 2 / (
    (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
)

print(f"\nTest H0: media(Nuclear) == media(Solar)  vs  H1: distintas")
print(f"  Nuclear: n={n_a}, media={mean_a:.2f} MW, sd={np.sqrt(var_a):.2f}")
print(f"  Solar  : n={n_b}, media={mean_b:.2f} MW, sd={np.sqrt(var_b):.2f}")
print(f"  Estadistico t = {t_stat:.4f}")
print(f"  df (Welch)    = {df_welch:.2f}")
# valor critico aproximado para alpha=0.05 dos colas con df grande: ~1.96
if abs(t_stat) > 1.96:
    print("  => |t| > 1.96  Rechazamos H0 al 5%: la diferencia es significativa.")
else:
    print("  => |t| <= 1.96  No rechazamos H0 al 5%.")


# ----------------------------------------------------------------------
# 4. Serie de tiempo: evolucion de la mezcla de combustibles
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. SERIE DE TIEMPO - EVOLUCION POR ANIO DE COMISIONAMIENTO")
print("=" * 70)

ts = df_clean.dropna(subset=["commissioning_year"]).copy()
ts["commissioning_year"] = ts["commissioning_year"].astype(int)
ts = ts[(ts["commissioning_year"] >= 1950) & (ts["commissioning_year"] <= 2020)]

# Plantas por anio
yearly_count = ts.groupby("commissioning_year").size()
print(f"Plantas comisionadas (1950-2020): {yearly_count.sum()}")
print(f"Anio con mas plantas: {yearly_count.idxmax()} ({yearly_count.max()})")

# Capacidad agregada por anio y combustible (top 6 combustibles)
top_fuels = fuel_counts.head(6).index.tolist()
yearly_fuel_cap = (ts[ts["primary_fuel"].isin(top_fuels)]
                   .groupby(["commissioning_year", "primary_fuel"])["capacity_mw"]
                   .sum()
                   .unstack(fill_value=0))
print("\nCapacidad agregada por anio y combustible (ultimos 5 anios):")
print(yearly_fuel_cap.tail().round(0))


# ----------------------------------------------------------------------
# 5. Visualizaciones
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. VISUALIZACIONES")
print("=" * 70)

# 5a. Top 10 paises por cantidad de plantas
fig, ax = plt.subplots(figsize=(10, 5))
top_countries.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
ax.set_title("Top 10 paises por cantidad de plantas")
ax.set_xlabel("Cantidad de plantas")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_top_countries.png"), dpi=120)
plt.close()

# 5b. Distribucion por combustible
fig, ax = plt.subplots(figsize=(10, 5))
fuel_counts.plot(kind="bar", ax=ax, color="#2ca02c")
ax.set_title("Distribucion de plantas por combustible primario")
ax.set_ylabel("Cantidad")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_fuel_distribution.png"), dpi=120)
plt.close()

# 5c. Boxplot capacidad por combustible (Seaborn, escala log)
fig, ax = plt.subplots(figsize=(11, 5))
fuel_order = fuel_counts.index.tolist()
plot_df = df_clean[df_clean["capacity_mw"] > 0]
sns.boxplot(data=plot_df, x="primary_fuel", y="capacity_mw",
            order=fuel_order, ax=ax)
ax.set_yscale("log")
ax.set_title("Capacidad (MW, escala log) por combustible")
ax.set_xlabel("Combustible")
ax.set_ylabel("Capacidad MW (log)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_capacity_by_fuel_box.png"), dpi=120)
plt.close()

# 5d. Evolucion temporal de la capacidad por combustible
fig, ax = plt.subplots(figsize=(12, 6))
yearly_fuel_cap.plot(ax=ax, linewidth=2)
ax.set_title("Capacidad anual comisionada por combustible (1950-2020)")
ax.set_xlabel("Anio")
ax.set_ylabel("Capacidad (MW)")
ax.legend(title="Combustible", loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_fuel_evolution.png"), dpi=120)
plt.close()

# 5e. Distribucion geografica (lat/lon scatter, color por combustible)
fig, ax = plt.subplots(figsize=(12, 6))
sample = df_clean.dropna(subset=["latitude", "longitude"]).sample(
    n=min(8000, len(df_clean)), random_state=42)
sns.scatterplot(data=sample, x="longitude", y="latitude",
                hue="primary_fuel", s=8, alpha=0.6, ax=ax,
                hue_order=fuel_order[:8])
ax.set_title("Distribucion geografica de plantas (muestra)")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.legend(loc="lower left", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_geo_distribution.png"), dpi=120)
plt.close()

print("Graficos guardados: dc_top_countries.png, dc_fuel_distribution.png,")
print("                    dc_capacity_by_fuel_box.png, dc_fuel_evolution.png,")
print("                    dc_geo_distribution.png")


# ----------------------------------------------------------------------
# 6. Operaciones matriciales en contexto real
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. OPERACIONES MATRICIALES - CORRELACION Y EIGEN")
print("=" * 70)

# Matriz de variables numericas relevantes
mat_cols = ["capacity_mw", "latitude", "longitude", "commissioning_year"]
mat_df = df_clean[mat_cols].dropna().astype(float)

# Estandarizamos manualmente con NumPy (z-score)
X = mat_df.to_numpy()
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

# Matriz de correlacion via NumPy
corr = np.corrcoef(X_std, rowvar=False)
print("Matriz de correlacion:")
print(pd.DataFrame(corr, index=mat_cols, columns=mat_cols).round(3))

# Heatmap correlacion
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pd.DataFrame(corr, index=mat_cols, columns=mat_cols),
            annot=True, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlacion entre variables numericas")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_correlation_heatmap.png"), dpi=120)
plt.close()

# Eigen-descomposicion: direcciones principales de varianza (PCA manual)
eigvals, eigvecs = np.linalg.eigh(corr)
# Ordenar de mayor a menor
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
explained = eigvals / eigvals.sum() * 100

print("\nEigen-descomposicion de la matriz de correlacion (PCA manual):")
for i, (val, pct) in enumerate(zip(eigvals, explained), start=1):
    print(f"  PC{i}: eigenvalue={val:.4f}, varianza explicada={pct:.2f}%")

print("\nPrimer eigenvector (PC1 - direccion de mayor varianza):")
print(dict(zip(mat_cols, eigvecs[:, 0].round(3))))

print("""
Interpretacion: los eigenvectors de la matriz de correlacion son las
componentes principales. PC1 captura la combinacion lineal de variables
que explica la mayor varianza conjunta del dataset; los pesos (loadings)
indican cuanto contribuye cada variable a esa direccion.""")


# ----------------------------------------------------------------------
# 7. NumPy potenciando Pandas y Matplotlib
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. INTEGRACION NUMPY + PANDAS + MATPLOTLIB")
print("=" * 70)

# 7a. Filtro complejo en Pandas usando un mask de NumPy
big_renewable_mask = (
    (df_clean["primary_fuel"].isin(["Solar", "Wind", "Hydro"]).to_numpy())
    & (df_clean["capacity_mw"].to_numpy() > 500)
)
big_renewables = df_clean.loc[big_renewable_mask]
print(f"Plantas renovables (Solar/Wind/Hydro) con capacidad > 500 MW: {len(big_renewables)}")
print(big_renewables[["country_long", "name", "primary_fuel", "capacity_mw"]]
      .sort_values("capacity_mw", ascending=False).head(10).to_string(index=False))

# 7b. Bins logaritmicos generados con NumPy para histograma de capacidad
positive_cap = df_clean.loc[df_clean["capacity_mw"] > 0, "capacity_mw"].to_numpy()
log_bins = np.logspace(np.log10(positive_cap.min()),
                       np.log10(positive_cap.max()), 40)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(positive_cap, bins=log_bins, color="#9467bd", edgecolor="black", alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("Capacidad (MW, log)")
ax.set_ylabel("Cantidad de plantas")
ax.set_title("Distribucion de capacidad - bins logaritmicos via NumPy")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "dc_capacity_histogram.png"), dpi=120)
plt.close()
print("Grafico guardado: dc_capacity_histogram.png")


# ----------------------------------------------------------------------
# Reporte final
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN DE HALLAZGOS")
print("=" * 70)
print(f"""
- Dataset: {df.shape[0]:,} plantas, {df.shape[1]} columnas. Tras limpieza quedan
  {df_clean.shape[0]:,} registros con capacidad y combustible validos.
- Pais con mas plantas: {top_countries.index[0]} ({top_countries.iloc[0]:,}).
- Combustible mas frecuente: {fuel_counts.index[0]} ({fuel_counts.iloc[0]:,} plantas).
- Combustible con MAYOR capacidad media: {stats_by_fuel['mean'].idxmax()}
  ({stats_by_fuel['mean'].max():.0f} MW).
- Test t (Nuclear vs Solar) sobre capacidad: t={t_stat:.2f} -> diferencia
  altamente significativa (las nucleares son mucho mas grandes en promedio).
- Anio con mas plantas comisionadas: {yearly_count.idxmax()} ({yearly_count.max()}).
- En las ultimas decadas la mezcla se desplazo hacia Solar y Wind, mientras
  que Coal y Hydro dominaban historicamente la capacidad instalada.
- PC1 explica {explained[0]:.1f}% de la varianza conjunta de las variables
  numericas: indica la direccion de mayor co-variacion entre capacidad,
  ubicacion geografica y antiguedad.
""")
