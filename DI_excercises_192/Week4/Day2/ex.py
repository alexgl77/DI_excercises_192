"""
Week 4 - Day 2: Ejercicios de NumPy avanzado, Pandas y Matplotlib.
Algebra lineal, estadistica, fechas, hipotesis y visualizacion.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(42)
output_dir = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Exercise 1: Matrix Operations - determinante e inversa
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 1: Determinante e inversa de matriz 3x3")
print("=" * 60)

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)

print("Matriz A:")
print(A)
print(f"\nDeterminante: {det_A:.4f}")
print("\nInversa de A:")
print(np.round(inv_A, 4))

# Verificacion: A * A_inv ~ I
print("\nVerificacion A @ A_inv (debe ser identidad):")
print(np.round(A @ inv_A, 4))
print()


# ----------------------------------------------------------------------
# Exercise 2: Statistical Analysis - media, mediana, std
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 2: Estadisticos sobre 50 numeros aleatorios")
print("=" * 60)

data = np.random.rand(50) * 100  # entre 0 y 100
mean_val = np.mean(data)
median_val = np.median(data)
std_val = np.std(data)

print(f"Media     : {mean_val:.4f}")
print(f"Mediana   : {median_val:.4f}")
print(f"Desv std  : {std_val:.4f}")
print()


# ----------------------------------------------------------------------
# Exercise 3: Date Manipulation
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 3: Fechas de enero 2023 reformateadas")
print("=" * 60)

dates_jan = np.arange("2023-01-01", "2023-02-01", dtype="datetime64[D]")
dates_formatted = pd.to_datetime(dates_jan).strftime("%Y/%m/%d")

print("Primeras 5 fechas (formato original):", dates_jan[:5])
print("Primeras 5 fechas (formato YYYY/MM/DD):", list(dates_formatted[:5]))
print(f"Total de dias: {len(dates_jan)}")
print()


# ----------------------------------------------------------------------
# Exercise 4: Data Manipulation con NumPy + Pandas
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 4: DataFrame con seleccion condicional y agregaciones")
print("=" * 60)

df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
print("DataFrame:")
print(df.round(3))

print("\nFilas donde A > 0:")
print(df[df["A"] > 0].round(3))

print("\nSuma por columna:")
print(df.sum().round(3))
print("\nPromedio por columna:")
print(df.mean().round(3))
print()


# ----------------------------------------------------------------------
# Exercise 5: Image Representation
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 5: Imagenes como NumPy arrays")
print("=" * 60)
print("""
Una imagen es un array de NumPy:
- Grayscale: shape (H, W), valores 0-255 (o 0.0-1.0).
- Color RGB : shape (H, W, 3), un canal por R, G, B.
- Color RGBA: shape (H, W, 4), agrega canal alpha.

Cada pixel = una entrada del array. Operar pixel a pixel = operar el array.
""")

gray_image = np.array([
    [  0,  50, 100, 150, 200],
    [ 50, 100, 150, 200, 250],
    [100, 150, 200, 250, 200],
    [150, 200, 250, 200, 150],
    [200, 250, 200, 150, 100],
], dtype=np.uint8)

print("Matriz 5x5 grayscale (uint8):")
print(gray_image)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(gray_image, cmap="gray", vmin=0, vmax=255)
ax.set_title("Grayscale 5x5")
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ex5_grayscale.png"), dpi=120)
plt.close()
print("(grafico guardado: ex5_grayscale.png)")
print()


# ----------------------------------------------------------------------
# Exercise 6: Basic Hypothesis Testing
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 6: Test de hipotesis - efectividad del entrenamiento")
print("=" * 60)

productivity_before = np.random.normal(loc=50, scale=10, size=30)
productivity_after = productivity_before + np.random.normal(loc=5, scale=3, size=30)

# H0: la media de la diferencia (after - before) es 0 (no hay efecto).
# H1: la media de la diferencia es > 0 (el entrenamiento mejora la productividad).
diff = productivity_after - productivity_before
n = len(diff)
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)
se = std_diff / np.sqrt(n)
t_stat = mean_diff / se
# Valor critico aproximado para t de una cola con alpha=0.05 y df=29: ~1.699
t_crit = 1.699

print(f"Media de la diferencia : {mean_diff:.4f}")
print(f"Desv std de la diferencia: {std_diff:.4f}")
print(f"Estadistico t           : {t_stat:.4f}")
print(f"Valor critico (alpha=0.05, una cola): {t_crit}")

if t_stat > t_crit:
    print("=> Rechazamos H0: el entrenamiento SI mejora la productividad.")
else:
    print("=> No rechazamos H0: no hay evidencia suficiente del efecto.")
print()


# ----------------------------------------------------------------------
# Exercise 7: Complex Array Comparison
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 7: Comparacion elemento a elemento")
print("=" * 60)

arr1 = np.array([5, 12, 3, 8, 19, 7])
arr2 = np.array([4, 15, 3, 10, 17, 6])

greater = arr1 > arr2
print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 > arr2: {greater}")
print()


# ----------------------------------------------------------------------
# Exercise 8: Time Series Data Manipulation
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 8: Serie de tiempo 2023 - slicing por trimestres")
print("=" * 60)

ts_index = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
ts = pd.Series(np.random.randn(len(ts_index)).cumsum(), index=ts_index)

q1 = ts["2023-01":"2023-03"]
q2 = ts["2023-04":"2023-06"]
q3 = ts["2023-07":"2023-09"]
q4 = ts["2023-10":"2023-12"]

print(f"Q1 (Ene-Mar): {len(q1)} dias, media={q1.mean():.3f}")
print(f"Q2 (Abr-Jun): {len(q2)} dias, media={q2.mean():.3f}")
print(f"Q3 (Jul-Sep): {len(q3)} dias, media={q3.mean():.3f}")
print(f"Q4 (Oct-Dic): {len(q4)} dias, media={q4.mean():.3f}")
print()


# ----------------------------------------------------------------------
# Exercise 9: Data Conversion NumPy <-> DataFrame
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 9: Conversion NumPy <-> DataFrame")
print("=" * 60)

np_arr = np.arange(12).reshape(4, 3)
df_from_np = pd.DataFrame(np_arr, columns=["X", "Y", "Z"])
np_from_df = df_from_np.to_numpy()

print("Array NumPy original:")
print(np_arr)
print("\nDataFrame desde array:")
print(df_from_np)
print("\nArray reconstruido desde DataFrame:")
print(np_from_df)
print(f"\nIguales? {np.array_equal(np_arr, np_from_df)}")
print()


# ----------------------------------------------------------------------
# Exercise 10: Basic Visualization
# ----------------------------------------------------------------------
print("=" * 60)
print("Exercise 10: Visualizacion basica con Matplotlib")
print("=" * 60)

x = np.arange(50)
y = np.random.randn(50).cumsum()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, y, marker="o", linestyle="-", color="#1f77b4")
ax.set_title("Random walk de 50 pasos")
ax.set_xlabel("Paso")
ax.set_ylabel("Valor acumulado")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ex10_random_walk.png"), dpi=120)
plt.close()

print("(grafico guardado: ex10_random_walk.png)")
