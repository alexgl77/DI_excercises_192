"""
Genera exercise.ipynb a partir del contenido de exercise.py para subir a Google Colab.
"""

import json
import os


def md_cell(*lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


cells = [
    md_cell(
        "# Week 4 - Day 5 - Machine Learning Exercises",
        "",
        "Curso: Developers Institute - Bootcamp Data / ML",
        "Autor: Alex Goldbaum",
        "",
        "Ejercicios teoricos sobre formulacion de problemas ML, seleccion de features,",
        "estrategias de evaluacion y diseno de soluciones para distintos escenarios.",
    ),

    # ---------- Ejercicio 1 ----------
    md_cell(
        "## Ejercicio 1: Loan Default Prediction - Problema y datos",
        "",
        "### Problem Statement",
        "Desarrollar un modelo de **clasificacion binaria** que estime la probabilidad de",
        "que un solicitante de prestamo entre en **default** (incumplimiento de pago) dentro",
        "de un horizonte definido (ej. 12 meses despues de aprobado el credito).",
        "",
        "El modelo apoyara al area de riesgo crediticio para:",
        "- Aprobar/rechazar solicitudes de forma mas consistente.",
        "- Ajustar tasas de interes segun riesgo individual.",
        "- Reducir la tasa de morosidad y las perdidas esperadas.",
        "",
        "**Salida del modelo:** probabilidad `P(default)` en `[0, 1]`, mas una decision binaria",
        "con umbral calibrado segun el costo asimetrico de FP vs FN.",
        "",
        "### Datos necesarios",
        "1. **Personales / demograficos:** edad, estado civil, dependientes, educacion, situacion habitacional.",
        "2. **Financieros:** ingreso, gastos, DTI, antiguedad laboral, tipo de empleo, patrimonio.",
        "3. **Credito:** score (FICO/Equifax/Dicom), historial, mora previa, consultas recientes, utilizacion de tarjetas.",
        "4. **Prestamo:** monto, plazo, tasa, proposito, garantias, co-deudores.",
        "5. **Comportamiento:** historial de pago interno, patrones de uso de cuenta.",
        "",
        "### Fuentes",
        "- Sistema core bancario / CRM interno.",
        "- Buros de credito (Equifax, Experian, TransUnion, Dicom).",
        "- Sistemas de scoring externos.",
        "- Datos publicos (desempleo regional, indices economicos).",
        "- Verificaciones gubernamentales (SII, AFIP, etc.).",
        "- Open banking / PSD2 con consentimiento.",
        "",
        "### Consideraciones",
        "- Privacidad: GDPR / ley 19.628 (Chile).",
        "- Fairness: evitar discriminacion por genero, raza, edad.",
        "- Class imbalance: defaults suelen ser <10% -> SMOTE, class_weight, threshold tuning.",
    ),

    # ---------- Ejercicio 2 ----------
    md_cell(
        "## Ejercicio 2: Feature Selection (Kaggle Loan Prediction)",
        "",
        "Variables tipicas del dataset: `Loan_ID, Gender, Married, Dependents, Education,`",
        "`Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,`",
        "`Credit_History, Property_Area, Loan_Status`.",
        "",
        "### Features mas relevantes",
        "1. **`Credit_History`** *** — la senal mas fuerte; sola explica gran parte del accuracy baseline.",
        "2. **`ApplicantIncome + CoapplicantIncome`** — capacidad de pago; conviene crear `TotalIncome` y un ratio DTI.",
        "3. **`LoanAmount`** — mayor monto, mayor exposicion al riesgo.",
        "4. **`Loan_Amount_Term`** — plazos largos suman incertidumbre.",
        "5. **`Property_Area`** — urban/semiurban/rural cambia perfil de riesgo y valor de garantia.",
        "6. **`Education`, `Self_Employed`** — proxy de estabilidad de ingresos.",
        "7. **`Married`, `Dependents`** — carga financiera; secundarias.",
        "",
        "### Descartables",
        "- **`Loan_ID`**: identificador puro, sin senal.",
        "- **`Gender`**: aporta senal estadistica pero introduce riesgo legal y de fairness. Recomendado excluir.",
        "",
        "### Ingenieria de features sugerida",
        "- `TotalIncome = ApplicantIncome + CoapplicantIncome`",
        "- `DTI = LoanAmount / (TotalIncome * Loan_Amount_Term)`",
        "- `LoanAmount_log = log1p(LoanAmount)` para corregir skew.",
        "- Imputar `Credit_History` faltante con moda + flag `missing_credit`.",
    ),

    # ---------- Ejercicio 3 ----------
    md_cell(
        "## Ejercicio 3: Modelo, evaluacion y optimizacion",
        "",
        "### Modelos recomendados",
        "1. **Logistic Regression** — baseline interpretable; coeficientes traducibles a odds; cumple requisitos regulatorios.",
        "2. **Gradient Boosting (XGBoost / LightGBM / CatBoost)** — mejor rendimiento en tabular; maneja interacciones no lineales.",
        "3. **Modelo calibrado + reglas de negocio** — combinar score con reglas duras (rechazo automatico si `Credit_History=0`).",
        "",
        "### Pipeline de evaluacion",
        "1. Split estratificado train/test (80/20).",
        "2. Cross-validation estratificado (k=5) sobre train para hiperparametros.",
        "3. Hold-out final solo al cierre, una sola vez.",
        "4. Calibrar threshold con curva de costo (no necesariamente 0.5).",
        "",
        "### Metricas relevantes (dataset desbalanceado -> accuracy es enganosa)",
        "- **ROC-AUC**: agregada, robusta a desbalance.",
        "- **PR-AUC**: mas informativa con clases muy desbalanceadas.",
        "- **Recall** clase positiva: cuantos defaults reales detectamos.",
        "- **Precision**: de los rechazados, cuantos eran efectivamente malos.",
        "- **F1 / F-beta** (beta>1 si recall pesa mas).",
        "- **Confusion matrix** y **costo esperado** del negocio.",
        "- **Calibration plot**: que las probabilidades predichas correspondan a frecuencias reales.",
        "",
        "### Optimizacion",
        "- Grid/Random/Bayesian search sobre hiperparametros.",
        "- Class weights o resampling (SMOTE).",
        "- Feature engineering iterativo guiado por SHAP / permutation importance.",
        "- Monitoreo en produccion: drift de features y degradacion de AUC.",
    ),

    # ---------- Ejercicio 4 ----------
    md_cell(
        "## Ejercicio 4: Tipo de ML para cada escenario",
        "",
        "### 1) Predecir precios de acciones",
        "**Tipo:** Supervised Learning - Regresion (serie temporal).",
        "",
        "**Por que:** target continuo (precio futuro); existen datos historicos etiquetados con la respuesta real observada en `t+1`, `t+5`, etc.",
        "",
        "**Modelos candidatos:** ARIMA/SARIMA (baseline), Gradient Boosting con features tecnicos (RSI, MACD, medias moviles), LSTM o Transformers para secuencias largas.",
        "",
        "**Nota:** mercados ruidosos y casi-eficientes; las mejoras marginales son lo realista, cuidado con overfitting.",
        "",
        "---",
        "",
        "### 2) Organizar una biblioteca por generos",
        "**Tipo:** Unsupervised Learning - Clustering.",
        "",
        "**Por que:** no tenemos etiquetas de genero previas (si las tuvieramos seria supervisado). El objetivo es **descubrir** estructura: agrupar libros por similitud de contenido.",
        "",
        "**Modelos candidatos:** K-Means o Agglomerative Clustering sobre embeddings (TF-IDF o sentence-transformers), HDBSCAN para clusters de tamano variable, LDA para topics interpretables.",
        "",
        "---",
        "",
        "### 3) Robot navegando un laberinto",
        "**Tipo:** Reinforcement Learning (o busqueda clasica).",
        "",
        "**Por que:** agente que toma acciones secuenciales y recibe recompensa (llegar al objetivo, evitar muros). No hay dataset supervisado de `(estado, accion_optima)`.",
        "",
        "**Modelos candidatos:** Q-Learning / SARSA en espacios discretos, DQN si el estado es de alta dimension (pixeles).",
        "",
        "**Nota:** si el laberinto es conocido y estatico, **A\\*** o **BFS** encuentran la solucion optima sin necesidad de aprender. RL brilla cuando el entorno es estocastico, parcialmente observable o cambia.",
    ),

    # ---------- Ejercicio 5 ----------
    md_cell(
        "## Ejercicio 5: Estrategia de evaluacion para 3 tipos de modelos",
        "",
        "### A) Supervised - Clasificacion (ej. spam vs no-spam)",
        "**Estrategia:** split train/val/test estratificado + k-fold CV (k=5 o 10).",
        "",
        "**Metricas:** Accuracy (solo si balanceado), Precision/Recall, F1, ROC-AUC, PR-AUC, confusion matrix.",
        "",
        "**Desafios:** desbalance de clases, data leakage en CV, concept drift en produccion.",
        "",
        "---",
        "",
        "### B) Unsupervised - Clustering (ej. K-Means de clientes)",
        "**Estrategia:** no hay ground truth -> evaluacion indirecta.",
        "",
        "**Metricas internas:** Silhouette score, Davies-Bouldin, Calinski-Harabasz, Elbow method (WCSS vs k).",
        "",
        "**Metricas externas (si hay labels parciales):** Adjusted Rand Index, Normalized Mutual Information.",
        "",
        "**Cualitativo:** inspeccion humana de los clusters + estabilidad con diferentes seeds.",
        "",
        "**Desafios:** 'bueno' depende del objetivo, no hay ground truth; K-Means asume clusters esfericos; curse of dimensionality.",
        "",
        "---",
        "",
        "### C) Reinforcement Learning (ej. agente de videojuego)",
        "**Estrategia:** entrenar en entorno simulado, medir sobre N seeds en episodios completos.",
        "",
        "**Metricas:** cumulative reward por episodio (media + std sobre seeds), convergencia, sample efficiency, exploration vs exploitation (epsilon/entropia), success rate.",
        "",
        "**Desafios:** reward sparse, inestabilidad entre seeds, generalizacion a entornos distintos al de entrenamiento, costo computacional alto.",
    ),

    md_cell(
        "## Conclusion",
        "",
        "El ejercicio cubre el ciclo conceptual completo de un proyecto ML: definir el problema",
        "y los datos, elegir features con criterio tecnico y de negocio (incluyendo fairness),",
        "seleccionar familia de modelos apropiada al tipo de problema (supervisado, no supervisado,",
        "refuerzo), y disenar una estrategia de evaluacion que combine metricas tecnicas con",
        "consideraciones de negocio y monitoreo en produccion.",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


out_path = os.path.join(os.path.dirname(__file__), "exercise.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generado: {out_path}")
print(f"Tamano: {os.path.getsize(out_path)} bytes")
