"""
Week 6 - Day 1
Introducción a Deep Learning vs Machine Learning tradicional + ANNs + Polynomial Fitting

Ejercicios:
1. Tabla comparativa ML tradicional vs Deep Learning
2. Diagrama de una ANN simple (texto ASCII)
3. Generación y visualización de dataset con ruido
4. Ajuste de modelos polinómicos de distintos grados (overfitting)
5. Cross-validation para encontrar el grado óptimo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Ejercicio 1: Deep Learning vs Traditional Machine Learning
# ---------------------------------------------------------------------------

ML_VS_DL_TABLE = """
| Aspecto                    | Machine Learning tradicional                  | Deep Learning                                     |
|----------------------------|-----------------------------------------------|---------------------------------------------------|
| Feature Engineering        | Manual, requiere experto del dominio          | Automática, la red aprende las features           |
| Data Processing            | Funciona bien con datos estructurados/tabular | Brilla con datos no estructurados (img, texto)    |
| Scalability                | Rendimiento se estanca con mucho dato         | Mejora a medida que crece el dataset              |
| Pattern Discovery          | Patrones lineales / poco profundos            | Patrones jerárquicos y no lineales muy complejos  |
| Computational Requirements | CPU suele bastar, entrenamiento rápido        | Requiere GPU/TPU, entrenamiento costoso           |
"""

ML_VS_DL_USE_CASES = """
Problema mejor para ML tradicional:
    Predicción de churn (abandono) de clientes con datos tabulares de pocas columnas
    y miles -no millones- de registros. Un Random Forest o una regresión logística
    rinden igual o mejor que una red neuronal, son interpretables y más baratos.

Problema mejor para Deep Learning:
    Diagnóstico médico a partir de imágenes (ej. detección de neumonía en
    radiografías). Las CNN aprenden representaciones jerárquicas de píxeles que
    serían imposibles de "feature-engineerar" a mano.
"""

DL_UNSTRUCTURED_PARAGRAPH = """
Deep Learning destaca con datos no estructurados porque sus capas aprenden
representaciones jerárquicas de forma automática: las primeras capas detectan
patrones simples (bordes, fonemas, n-gramas) y las capas profundas combinan esos
patrones en conceptos abstractos (caras, objetos, intención semántica). En ML
tradicional ese paso requiere feature engineering manual que es lento, frágil y
limitado por el conocimiento del experto. Además, las redes profundas escalan
con los datos: a más imágenes/texto/audio mejor representación aprenden,
mientras que los modelos clásicos se saturan rápido. Por último, arquitecturas
como CNN, RNN y Transformers incorporan inductive biases (convolución,
recurrencia, atención) que se alinean con la estructura de los datos crudos.
"""


# ---------------------------------------------------------------------------
# Ejercicio 2: Artificial Neural Network (diagrama ASCII)
# ---------------------------------------------------------------------------

ANN_DIAGRAM = r"""
        Input layer        Hidden layer          Output layer
        (3 neuronas)       (4 neuronas)          (2 neuronas)

           x1 ●----w11---->●----.
                  \        ●----.\
           x2 ●----w22---->●----. \--> ● y1
                  \        ●----. /
           x3 ●----w33---->●----./--> ● y2

                  ↑           ↑              ↑
               pesos (w)   activación f()  salida

Componentes etiquetados:
- Neuronas (●): unidades de cómputo (entradas, ocultas y salidas).
- Pesos (w_ij): fuerza de la conexión entre neuronas de capas consecutivas.
- Bias (b): término aditivo por neurona que desplaza la activación.
- Función de activación f(): no linealidad aplicada a (Σ w·x + b),
    por ejemplo ReLU, sigmoide o tanh.
- Capas: input → hidden → output.

Flujo de información (forward pass):
1. Cada neurona oculta calcula z = Σ (w_ij · x_i) + b_j.
2. Se aplica la función de activación a = f(z), introduciendo no linealidad.
3. Las activaciones de la capa oculta se propagan a la capa de salida usando
   otra matriz de pesos y bias.
4. La capa de salida produce las predicciones (por ej. softmax para
   clasificación). Durante el entrenamiento, la pérdida se propaga hacia atrás
   (backpropagation) para ajustar pesos y biases con descenso de gradiente.
"""


# ---------------------------------------------------------------------------
# Ejercicio 3: Dataset con ruido
# ---------------------------------------------------------------------------

def build_dataset():
    """Genera 20 puntos con y = -x^2 + ruido gaussiano y separa train/test."""
    np.random.seed(0)
    x = np.arange(-1, 1, 0.1)
    y = -x ** 2 + np.random.normal(0, 0.05, len(x))

    x_train, y_train = x[:12], y[:12]
    x_test, y_test = x[12:], y[12:]
    return x, y, x_train, y_train, x_test, y_test


def plot_dataset(x, y):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="steelblue", label="Datos con ruido")
    plt.title("Dataset y = -x² + ruido N(0, 0.05)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ---------------------------------------------------------------------------
# Ejercicio 4: Polynomial fitting
# ---------------------------------------------------------------------------

def polynomial_fit(degree, x_train, y_train):
    """Devuelve un np.poly1d ajustado a los datos de entrenamiento."""
    coefs = np.polyfit(x_train, y_train, degree)
    return np.poly1d(coefs)


def plot_polyfit(degree, x_train, y_train, x_test, y_test):
    poly = polynomial_fit(degree, x_train, y_train)
    x_curve = np.linspace(-1, 1, 200)

    plt.figure(figsize=(7, 5))
    plt.scatter(x_train, y_train, color="steelblue", label="Train")
    plt.scatter(x_test, y_test, color="orangered", label="Test")
    plt.plot(x_curve, poly(x_curve), color="black",
             label=f"Polinomio grado {degree}")
    plt.title(f"Ajuste polinómico de grado {degree}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-1.5, 0.5)
    plt.show()


# ---------------------------------------------------------------------------
# Ejercicio 5: Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_degrees(x_train, y_train, x_test, y_test, max_degree=11):
    results = []
    for d in range(1, max_degree + 1):
        poly = polynomial_fit(d, x_train, y_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, poly(x_train)))
        rmse_test = np.sqrt(mean_squared_error(y_test, poly(x_test)))
        results.append((d, rmse_train, rmse_test))
    return results


def plot_rmse_curve(results):
    degrees = [r[0] for r in results]
    rmse_train = [r[1] for r in results]
    rmse_test = [r[2] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(degrees, rmse_train, "o-", label="RMSE Train")
    plt.plot(degrees, rmse_test, "s-", label="RMSE Test")
    plt.yscale("log")
    plt.xlabel("Grado del polinomio")
    plt.ylabel("RMSE (escala log)")
    plt.title("RMSE vs grado del polinomio")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EJERCICIO 1 - ML tradicional vs Deep Learning")
    print("=" * 70)
    print(ML_VS_DL_TABLE)
    print(ML_VS_DL_USE_CASES)
    print(DL_UNSTRUCTURED_PARAGRAPH)

    print("=" * 70)
    print("EJERCICIO 2 - Artificial Neural Network")
    print("=" * 70)
    print(ANN_DIAGRAM)

    print("=" * 70)
    print("EJERCICIO 3 - Dataset con ruido")
    print("=" * 70)
    x, y, x_train, y_train, x_test, y_test = build_dataset()
    print(f"Total puntos: {len(x)} | Train: {len(x_train)} | Test: {len(x_test)}")
    plot_dataset(x, y)

    print("=" * 70)
    print("EJERCICIO 4 - Polynomial fitting (grados 1, 7, 11)")
    print("=" * 70)
    for d in (1, 7, 11):
        plot_polyfit(d, x_train, y_train, x_test, y_test)

    print("=" * 70)
    print("EJERCICIO 5 - Cross-validation")
    print("=" * 70)
    results = cross_validate_degrees(x_train, y_train, x_test, y_test)
    print(f"{'Degree':>6} | {'RMSE Train':>12} | {'RMSE Test':>12}")
    print("-" * 38)
    for d, rt, rv in results:
        print(f"{d:>6} | {rt:>12.5f} | {rv:>12.5f}")

    best = min(results, key=lambda r: r[2])
    print(
        f"\nGrado óptimo (mín. RMSE test): {best[0]} "
        f"-> coincide con el modelo verdadero y = -x² (grado 2)."
    )
    plot_rmse_curve(results)


if __name__ == "__main__":
    main()
