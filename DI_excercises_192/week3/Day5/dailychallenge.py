"""
Week 3 - Day 5 - Daily Challenge
Interactive Data Visualization with Matplotlib and Seaborn
Dataset: US Superstore

Estructura:
  1. Carga y limpieza del dataset
  2. Matplotlib: line chart (ventas por año) + "mapa" por estado
  3. Seaborn: top 10 productos por sales + scatter profit vs discount
  4. Análisis comparativo Matplotlib vs Seaborn (comentarios al final)
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\alexg\Downloads\superstore\Sample - Superstore.csv"


# ============================================================
# 1. Data Preparation
# ============================================================

def load_and_clean(path):
    df = pd.read_csv(path, encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Order Year"] = df["Order Date"].dt.year

    df = df.drop_duplicates()
    df = df.dropna(subset=["Sales", "Profit", "Discount"])

    return df


df = load_and_clean(DATASET_PATH)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Rango de fechas: {df['Order Date'].min().date()} a {df['Order Date'].max().date()}")
print(f"Países en el dataset: {df['Country'].unique()}")
print()


# ============================================================
# 2. Matplotlib — Visualizaciones
# ============================================================

# 2a) Line chart: ventas a lo largo de los años (mensual)
# "Interactivo" en Matplotlib se logra via backend (zoom, pan, hover sobre puntos).
# Aquí dejamos la figura lista; al ejecutar el script con `plt.show()`
# la ventana incluye toolbar de Matplotlib (zoom, pan, save).

monthly_sales = (
    df.set_index("Order Date")
      .resample("ME")["Sales"]
      .sum()
      .reset_index()
)

fig1, ax1 = plt.subplots(figsize=(13, 5))
ax1.plot(monthly_sales["Order Date"], monthly_sales["Sales"],
         color="steelblue", linewidth=1.8, marker="o", markersize=4)
ax1.set_title("Tendencia de ventas mensuales (2014–2018)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Sales ($)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(SCRIPT_DIR, "dc_matplotlib_sales_trend.png"))
print("Guardado: dc_matplotlib_sales_trend.png")


# 2b) "Mapa" de ventas por área geográfica.
# El dataset solo contiene un país (United States), así que el equivalente
# significativo de "mapa por country" es un breakdown por State.
state_sales = (
    df.groupby("State")["Sales"]
      .sum()
      .sort_values(ascending=True)
)

fig2, ax2 = plt.subplots(figsize=(10, 12))
colors = ["seagreen" if v > state_sales.median() else "steelblue" for v in state_sales.values]
ax2.barh(state_sales.index, state_sales.values, color=colors, edgecolor="white")
ax2.set_title("Distribución geográfica de ventas — US (por estado)",
              fontsize=13, fontweight="bold")
ax2.set_xlabel("Total Sales ($)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
fig2.tight_layout()
fig2.savefig(os.path.join(SCRIPT_DIR, "dc_matplotlib_state_map.png"))
print("Guardado: dc_matplotlib_state_map.png")


# ============================================================
# 3. Seaborn — Visualizaciones
# ============================================================

# 3a) Top 10 productos por ventas
top10_products = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

fig3, ax3 = plt.subplots(figsize=(11, 6))
sns.barplot(data=top10_products, x="Sales", y="Product Name",
            palette="viridis", ax=ax3)
ax3.set_title("Top 10 productos por ventas totales", fontsize=14, fontweight="bold")
ax3.set_xlabel("Total Sales ($)")
ax3.set_ylabel("")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
fig3.tight_layout()
fig3.savefig(os.path.join(SCRIPT_DIR, "dc_seaborn_top10_products.png"))
print("Guardado: dc_seaborn_top10_products.png")


# 3b) Scatter: Profit vs Discount, segmentado por Category
fig4, ax4 = plt.subplots(figsize=(11, 6))
sns.scatterplot(data=df, x="Discount", y="Profit",
                hue="Category", alpha=0.45, s=30, ax=ax4)
ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax4.set_title("Relación entre descuento y profit (por categoría)",
              fontsize=14, fontweight="bold")
ax4.set_xlabel("Discount")
ax4.set_ylabel("Profit ($)")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax4.legend(title="Category")
fig4.tight_layout()
fig4.savefig(os.path.join(SCRIPT_DIR, "dc_seaborn_profit_vs_discount.png"))
print("Guardado: dc_seaborn_profit_vs_discount.png")


# ============================================================
# 4. Análisis comparativo + insights
# ============================================================

# Correlación numérica profit vs discount
corr = df["Discount"].corr(df["Profit"])
loss_rate_high_disc = (df[df["Discount"] >= 0.3]["Profit"] < 0).mean() * 100

print("\n" + "=" * 60)
print("INSIGHTS")
print("=" * 60)
print(f"- Correlacion Discount/Profit: {corr:.3f} (negativa -> mas descuento = menos profit)")
print(f"- En transacciones con descuento >=30%, el {loss_rate_high_disc:.1f}% pierde dinero.")

top_states_share = state_sales.tail(5).sum() / state_sales.sum() * 100
print(f"- Los 5 estados con más ventas concentran el {top_states_share:.1f}% del total.")

peak_month = monthly_sales.loc[monthly_sales["Sales"].idxmax(), "Order Date"]
print(f"- Pico mensual de ventas: {peak_month.strftime('%B %Y')} "
      f"(${monthly_sales['Sales'].max():,.0f}).")


# ============================================================
# Comparativa Matplotlib vs Seaborn
# ============================================================
# Matplotlib:
#   + Control fino: cada elemento (color, label, axis, formatter) es configurable.
#   + Ideal para gráficos custom y para integrar con Tkinter/IPython (interactividad).
#   - Verboso: requiere más código para resultados visualmente prolijos.
#
# Seaborn:
#   + Estética por defecto mucho mejor, con menos código.
#   + Excelente para gráficos estadísticos (scatter con hue, distribuciones, etc.).
#   + Integra de forma natural con DataFrames de pandas.
#   - Para casos extremos de customización suele tocar bajar a Matplotlib igual.
#
# Recomendación práctica:
#   - Exploración rápida y análisis estadístico → Seaborn.
#   - Dashboards finales, visuales con branding o interactividad → Matplotlib (+ ipywidgets/mpl_interactions).


if __name__ == "__main__":
    # Cuando se ejecuta directamente, mostrar las figuras (modo interactivo).
    # Si se corre en un entorno sin display, las imágenes ya están guardadas como PNG.
    try:
        plt.show()
    except Exception:
        pass
