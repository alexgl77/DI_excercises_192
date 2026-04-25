"""
Daily Challenge: NumPy, Pandas, and Matplotlib Integration

Analizar tendencias de temperatura mensual para 10 ciudades durante 12 meses.
Generamos los datos con NumPy, los manipulamos con Pandas y visualizamos con Matplotlib.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Data Preparation
# ----------------------------------------------------------------------

# Reproducibilidad
np.random.seed(42)

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
cities = [f"City {i}" for i in range(1, 11)]

# Temperaturas entre -5 y 35 grados, shape (10 ciudades, 12 meses)
temperatures = np.random.uniform(-5, 35, size=(len(cities), len(months)))

df = pd.DataFrame(temperatures, index=cities, columns=months)

print("Temperaturas mensuales por ciudad (degC):")
print(df.round(2))
print()


# ----------------------------------------------------------------------
# 2. Data Analysis
# ----------------------------------------------------------------------

# Promedio anual por ciudad (axis=1 promedia a lo largo de los meses)
annual_avg = df.mean(axis=1)

hottest_city = annual_avg.idxmax()
coldest_city = annual_avg.idxmin()

print("Promedio anual por ciudad (degC):")
print(annual_avg.round(2))
print()
print(f"Ciudad mas calida: {hottest_city} ({annual_avg.max():.2f} degC)")
print(f"Ciudad mas fria : {coldest_city} ({annual_avg.min():.2f} degC)")
print()

# Mes mas calido y mas frio en promedio global
monthly_avg = df.mean(axis=0)
print("Promedio mensual global (degC):")
print(monthly_avg.round(2))
print()
print(f"Mes mas calido: {monthly_avg.idxmax()} ({monthly_avg.max():.2f} degC)")
print(f"Mes mas frio : {monthly_avg.idxmin()} ({monthly_avg.min():.2f} degC)")
print()


# ----------------------------------------------------------------------
# 3. Data Visualization
# ----------------------------------------------------------------------

output_dir = os.path.dirname(os.path.abspath(__file__))

# Grafico 1: tendencia mensual de cada ciudad
fig, ax = plt.subplots(figsize=(12, 6))
for city in df.index:
    ax.plot(df.columns, df.loc[city], marker="o", label=city)
ax.set_title("Monthly Temperature Trends for Each City")
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (degC)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower left", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "monthly_temperature_trends.png"), dpi=120)
plt.close()

# Grafico 2: ranking de promedio anual por ciudad
fig, ax = plt.subplots(figsize=(10, 5))
sorted_avg = annual_avg.sort_values()
colors = ["#1f77b4"] * len(sorted_avg)
colors[0] = "#2ca02c"   # ciudad mas fria en verde
colors[-1] = "#d62728"  # ciudad mas calida en rojo
ax.barh(sorted_avg.index, sorted_avg.values, color=colors)
ax.set_title("Average Annual Temperature by City")
ax.set_xlabel("Temperature (degC)")
ax.grid(True, axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "annual_average_by_city.png"), dpi=120)
plt.close()

# Grafico 3: heatmap de la matriz completa
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(df.values, cmap="coolwarm", aspect="auto")
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns)
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)
ax.set_title("Temperature Heatmap (City x Month)")
fig.colorbar(im, ax=ax, label="Temperature (degC)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "temperature_heatmap.png"), dpi=120)
plt.close()


# ----------------------------------------------------------------------
# 4. Reporte breve
# ----------------------------------------------------------------------

report = f"""
============================================================
REPORTE: Tendencias de Temperatura
============================================================

Dataset sintetico generado con NumPy: 10 ciudades x 12 meses,
temperaturas uniformes entre -5 y 35 degC.

Hallazgos principales:
- Ciudad mas calida (promedio anual): {hottest_city} con {annual_avg.max():.2f} degC.
- Ciudad mas fria (promedio anual) : {coldest_city} con {annual_avg.min():.2f} degC.
- Diferencia entre ambas: {annual_avg.max() - annual_avg.min():.2f} degC.
- Mes mas calido (promedio global): {monthly_avg.idxmax()} ({monthly_avg.max():.2f} degC).
- Mes mas frio (promedio global) : {monthly_avg.idxmin()} ({monthly_avg.min():.2f} degC).

Observaciones:
- Como los datos son uniformes aleatorios, no aparece un patron estacional
  claro (verano/invierno). En un dataset real esperariamos ver maximos en
  los meses de verano y minimos en invierno segun el hemisferio.
- La dispersion entre ciudades es alta porque cada serie es independiente,
  por eso el grafico de lineas se ve "ruidoso". Esto refleja el caracter
  sintetico del dataset.

Graficos generados:
- monthly_temperature_trends.png
- annual_average_by_city.png
- temperature_heatmap.png
============================================================
"""
print(report)
