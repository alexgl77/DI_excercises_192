"""
Week 4 - Day 5 - Exercises
Ejercicios teoricos sobre Machine Learning:
  1. Definicion del problema y recoleccion de datos (Loan Default)
  2. Seleccion de features y eleccion del modelo
  3. Entrenamiento, evaluacion y optimizacion del modelo
  4. Diseno de soluciones ML para tres escenarios
  5. Estrategia de evaluacion para 3 tipos de modelos (supervisado / no supervisado / refuerzo)

Este archivo esta estructurado como documento ejecutable: cada ejercicio
es una funcion que imprime su contenido. Se ejecuta con `python exercise.py`.
"""


def line(char="=", n=70):
    print(char * n)


# ============================================================
# EJERCICIO 1
# ============================================================
def exercise_1():
    line()
    print("EJERCICIO 1: Loan Default Prediction - Problema y datos")
    line()

    print("""
PROBLEM STATEMENT
-----------------
Desarrollar un modelo de clasificacion binaria que estime la probabilidad
de que un solicitante de prestamo entre en default (incumplimiento de pago)
dentro de un horizonte definido (ej. 12 meses despues de aprobado el credito).
El modelo apoyara al area de riesgo crediticio para:
  - Aprobar/rechazar solicitudes de forma mas consistente.
  - Ajustar tasas de interes segun riesgo individual.
  - Reducir la tasa de morosidad y las perdidas esperadas.

Salida del modelo: probabilidad P(default) en [0, 1], y una decision binaria
con umbral calibrado segun el costo de falso negativo vs falso positivo.

DATOS QUE SE NECESITAN
----------------------
1) Personales / demograficos:
   - Edad, estado civil, dependientes, nivel educacional, situacion habitacional.
   - Importante manejar variables sensibles con cuidado (sesgo, fairness, regulacion).

2) Financieros:
   - Ingreso mensual, gastos fijos, ratio deuda/ingreso (DTI).
   - Antiguedad laboral, tipo de empleo, historial de empleos.
   - Patrimonio (cuentas, inversiones, propiedades).

3) Credito:
   - Score crediticio (FICO, Equifax, etc.).
   - Historial de creditos previos: cantidad, montos, mora historica.
   - Numero de consultas recientes al buro.
   - Utilizacion de tarjetas de credito.

4) Detalles del prestamo:
   - Monto solicitado, plazo, tasa, proposito (auto, hipotecario, consumo).
   - Garantias (colateral) y co-deudores.

5) Comportamiento de pago:
   - Historial de pagos anteriores en la misma institucion.
   - Patrones de uso de cuenta corriente / debito.

FUENTES DE DATOS
----------------
  - Registros internos del banco (CRM, sistema core bancario).
  - Buros de credito (Equifax, Experian, TransUnion, Dicom en Chile).
  - Sistemas de scoring externos.
  - Datos publicos: tasa de desempleo regional, indices economicos.
  - Verificaciones contra organismos gubernamentales (AFIP, SII, etc.).
  - Open banking / PSD2 (con consentimiento del cliente).

CONSIDERACIONES IMPORTANTES
---------------------------
  - Privacidad y proteccion de datos (GDPR, ley 19.628 en Chile).
  - Bias y fairness: no discriminar por genero, raza, edad de forma indebida.
  - Imbalance de clases: defaults suelen ser <10% de la poblacion -> tecnicas
    como SMOTE, class_weight, o ajuste de threshold.
""")


# ============================================================
# EJERCICIO 2
# ============================================================
def exercise_2():
    line()
    print("EJERCICIO 2: Feature Selection")
    line()

    print("""
Dataset referencia: Kaggle Loan Prediction.
Variables tipicas: Loan_ID, Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status.

FEATURES MAS RELEVANTES (con justificacion)
-------------------------------------------
1) Credit_History  ***
   - Es la senal mas fuerte: un historial limpio reduce drasticamente la
     probabilidad de default. En la mayoria de modelos baseline de este
     dataset, esta variable sola explica gran parte del accuracy.

2) ApplicantIncome + CoapplicantIncome
   - Capacidad de pago. Conviene sumarlos en una feature TotalIncome o
     calcular un ratio LoanAmount / TotalIncome (DTI).

3) LoanAmount
   - Monto solicitado: mas grande, mayor exposicion al riesgo.

4) Loan_Amount_Term
   - Plazo: prestamos largos tienen mas incertidumbre.

5) Property_Area
   - Urban / Semiurban / Rural cambia el perfil de riesgo y los precios
     de garantia.

6) Education y Self_Employed
   - Proxy de estabilidad de ingresos. Self-employed historicamente tiene
     mayor varianza.

7) Married y Dependents
   - Ayudan a estimar carga financiera; suelen ser secundarias.

FEATURES MENOS UTILES / DESCARTABLES
------------------------------------
  - Loan_ID: identificador puro, no aporta senal predictiva.
  - Gender: puede aportar senal estadistica pero introduce riesgo legal
    y de fairness (discriminacion). Recomendado excluir o auditar con
    cuidado.

INGENIERIA DE FEATURES SUGERIDA
-------------------------------
  - TotalIncome = ApplicantIncome + CoapplicantIncome
  - DTI = LoanAmount / (TotalIncome * Loan_Amount_Term) (capacidad de pago)
  - LoanAmount_log = log1p(LoanAmount) para corregir skewness.
  - Imputar Credit_History faltante con moda y crear flag "missing_credit".
""")


# ============================================================
# EJERCICIO 3
# ============================================================
def exercise_3():
    line()
    print("EJERCICIO 3: Modelo, evaluacion y optimizacion")
    line()

    print("""
MODELOS RECOMENDADOS (en orden de prioridad)
--------------------------------------------
1) Logistic Regression
   - Baseline interpretable; coeficientes traducibles a "odds".
   - Cumple bien requisitos regulatorios (banca exige explicabilidad).

2) Random Forest / Gradient Boosting (XGBoost, LightGBM, CatBoost)
   - Maneja interacciones no lineales y variables categoricas con menos
     preprocesamiento. Suele dar el mejor rendimiento bruto en tabular.

3) Modelos calibrados + reglas de negocio
   - Combinar el score del modelo con reglas duras (ej. rechazo automatico
     si Credit_History = 0). Importante en produccion bancaria.

PIPELINE DE EVALUACION
----------------------
  1. Split estratificado train/test (80/20) preservando proporcion de defaults.
  2. Cross-validation estratificado (k=5) sobre train para ajustar hiperparametros.
  3. Hold-out final solo se toca al final, una sola vez.
  4. Calibrar el threshold con la curva de costo (no necesariamente 0.5).

METRICAS RELEVANTES
-------------------
Como el dataset esta desbalanceado, accuracy es enganosa. Lo correcto:

  - ROC-AUC: medida agregada, robusta a desbalance.
  - PR-AUC (Precision-Recall AUC): mas informativa con clases muy desbalanceadas.
  - Recall en la clase positiva (default = 1): cuantos defaults reales detectamos.
    En banca, perder un default suele costar mucho mas que un falso rechazo.
  - Precision: de los rechazados, cuantos eran efectivamente malos.
  - F1 / F-beta (beta>1 si recall pesa mas).
  - Confusion matrix completa y costo esperado (matriz de costos del negocio).
  - Calibration plot: que P(default) prediga frecuencias reales.

OPTIMIZACION
------------
  - Grid / Random / Bayesian search sobre hiperparametros (max_depth,
    learning_rate, n_estimators, regularizacion).
  - Class weights o resampling (SMOTE) para el desbalance.
  - Feature engineering iterativo guiado por SHAP / permutation importance.
  - Monitoreo en produccion: drift de features y degradacion de AUC.
""")


# ============================================================
# EJERCICIO 4
# ============================================================
def exercise_4():
    line()
    print("EJERCICIO 4: Tipo de ML para cada escenario")
    line()

    print("""
1) PREDECIR PRECIOS DE ACCIONES
   Tipo: SUPERVISED LEARNING - Regresion (serie temporal).
   Justificacion:
     - Target continuo (precio futuro) -> regresion.
     - Datos historicos etiquetados con la "respuesta correcta" (precio real
       observado en t+1, t+5, etc.).
   Modelos candidatos:
     - ARIMA / SARIMA para baseline univariado.
     - Gradient Boosting con features tecnicos (RSI, MACD, medias moviles).
     - LSTM / Transformers para secuencias largas.
   Nota: los mercados son ruidosos y casi-eficientes, asi que las mejoras
   marginales sobre un baseline son lo realista; cuidado con sobreajuste.

2) ORGANIZAR UNA BIBLIOTECA POR GENEROS / SIMILITUD
   Tipo: UNSUPERVISED LEARNING - Clustering.
   Justificacion:
     - No tenemos etiquetas de genero previas (si las tuvieramos seria un
       problema de clasificacion supervisada).
     - El objetivo es DESCUBRIR estructura: agrupar libros por similitud
       de contenido (TF-IDF / embeddings sobre sinopsis o texto completo).
   Modelos candidatos:
     - K-Means o Agglomerative Clustering sobre embeddings.
     - HDBSCAN si esperamos clusters de tamanos distintos.
     - LDA (Latent Dirichlet Allocation) si queremos topics interpretables.

3) ROBOT NAVEGANDO UN LABERINTO - CAMINO MAS CORTO
   Tipo: REINFORCEMENT LEARNING (o busqueda clasica).
   Justificacion:
     - Hay un agente que toma acciones secuenciales y recibe una recompensa
       (llegar al objetivo, evitar muros). No hay un dataset de pares
       (estado, accion_optima) supervisado.
     - El agente aprende por prueba y error a maximizar la recompensa total.
   Modelos candidatos:
     - Q-Learning / SARSA para espacios discretos.
     - Deep Q-Network (DQN) si el estado es de alta dimension (ej. pixeles).
   Nota: si el laberinto es conocido y estatico, un algoritmo CLASICO
   como A* o BFS encuentra la solucion optima sin necesidad de aprender.
   RL brilla cuando el entorno es estocastico, parcialmente observable o cambia.
""")


# ============================================================
# EJERCICIO 5
# ============================================================
def exercise_5():
    line()
    print("EJERCICIO 5: Estrategia de evaluacion para 3 tipos de modelos")
    line()

    print("""
A) MODELO SUPERVISADO - Clasificacion (ej. spam vs no-spam)
   Estrategia:
     - Split train/val/test estratificado.
     - K-fold cross-validation (k=5 o 10) para reducir varianza.
   Metricas:
     - Accuracy: solo si las clases estan balanceadas.
     - Precision / Recall: cuando el costo de FP y FN es asimetrico.
     - F1-score: balance entre precision y recall.
     - ROC-AUC y PR-AUC: medidas agregadas independientes del threshold.
     - Confusion matrix: para entender errores por clase.
   Desafios:
     - Desbalance de clases.
     - Data leakage en cross-validation (features que filtran info del futuro).
     - Cambios de distribucion (concept drift) en produccion.

B) MODELO NO SUPERVISADO - Clustering (ej. K-Means de clientes)
   Estrategia:
     - No hay etiquetas verdaderas, asi que la evaluacion es indirecta.
   Metricas internas:
     - Silhouette score: cohesion intra-cluster vs separacion inter-cluster.
     - Davies-Bouldin: menor es mejor.
     - Calinski-Harabasz: mayor es mejor.
     - Elbow method (inertia / WCSS vs k) para elegir numero de clusters.
   Metricas externas (si hay etiquetas parciales):
     - Adjusted Rand Index, Normalized Mutual Information.
   Validacion cualitativa:
     - Inspeccion humana de los clusters: tienen sentido de negocio?
     - Estabilidad: con diferentes seeds / subsamples se obtienen clusters parecidos?
   Desafios:
     - "Bueno" depende del objetivo; no hay ground truth.
     - K-Means asume clusters esfericos similares; en datos reales suele fallar.
     - Curse of dimensionality: en alta dimension las distancias colapsan.

C) MODELO REINFORCEMENT LEARNING (ej. agente que juega un videojuego)
   Estrategia:
     - Entrenar en un entorno simulado (gym, MuJoCo, simulador propio).
     - Medir desempeno en episodios completos sobre N seeds distintos.
   Metricas:
     - Cumulative reward por episodio (promedio y desviacion sobre seeds).
     - Convergencia: cuantos episodios hasta estabilizar el reward.
     - Sample efficiency: reward / numero de interacciones con el entorno.
     - Exploration vs exploitation: tasa de exploracion (epsilon, entropia)
       y como decae.
     - Win rate / success rate si la tarea tiene metrica binaria de exito.
   Desafios:
     - Reward sparse: poca senal de aprendizaje (ej. solo recompensa al
       terminar). Requiere reward shaping o curriculum learning.
     - Inestabilidad del entrenamiento (varianza alta entre seeds).
     - Generalizacion: un agente que va bien en un entorno suele fallar en
       otro ligeramente distinto (overfitting al entorno de entrenamiento).
     - Costo computacional: millones de interacciones para entornos complejos.
""")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
