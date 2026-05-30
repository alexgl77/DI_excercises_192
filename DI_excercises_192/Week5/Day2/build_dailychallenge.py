"""
Generate dailychallenge.ipynb for Week 5 Day 2: Breast Cancer prediction.
Four classifiers: Logistic Regression, KNN, Random Forest, SVM.
Dataset: UCI Wisconsin Diagnostic Breast Cancer (569 rows).
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
        "# Daily Challenge: Breast Cancer Prediction",
        "",
        "**Course:** Developers Institute  **Week 5 - Day 2**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Goal: predict whether a breast mass is **malignant** (M) or **benign** (B) from",
        "30 features computed on digitized images of fine-needle aspirates. We compare",
        "four classifiers — Logistic Regression, K-Nearest Neighbors, Random Forest, and",
        "SVM — and pick the best one on the test set.",
        "",
        "Dataset: UCI Wisconsin Diagnostic Breast Cancer (569 samples, 30 features).",
    ),

    md("## Setup"),
    code(
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.neighbors import KNeighborsClassifier",
        "from sklearn.ensemble import RandomForestClassifier",
        "from sklearn.svm import SVC",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay",
        "",
        "sns.set_theme(style='whitegrid')",
        "RANDOM_STATE = 42",
    ),

    # --- 1. EDA ---
    md(
        "## 1. Exploratory Data Analysis",
        "",
        "### 1.1 Load and inspect",
    ),
    code(
        "# UCI mirror — same dataset that ships on Kaggle as `data.csv`",
        "URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'",
        "",
        "# Build the column names matching the standard Wisconsin Diagnostic format",
        "feature_names = [",
        "    'radius', 'texture', 'perimeter', 'area', 'smoothness',",
        "    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension',",
        "]",
        "stat_suffixes = ['_mean', '_se', '_worst']",
        "columns = ['id', 'diagnosis'] + [f + s for s in stat_suffixes for f in feature_names]",
        "",
        "df = pd.read_csv(URL, header=None, names=columns)",
        "print('Shape:', df.shape)",
        "df.head()",
    ),
    code(
        "# Quick info + dtypes",
        "df.info()",
    ),
    md("### 1.2 Check and handle missing values"),
    code(
        "missing = df.isna().sum()",
        "print('Columns with missing values:')",
        "print(missing[missing > 0] if missing.sum() > 0 else '  (no missing values)')",
        "print(f'\\nTotal missing cells: {df.isna().sum().sum()}')",
    ),
    md(
        "**Observation.** The Wisconsin Diagnostic CSV has no missing values, so no",
        "imputation is required. (If the Kaggle version with an empty `Unnamed: 32`",
        "column were used, we would drop it here.)",
    ),
    md("### 1.3 Drop any unnecessary column"),
    code(
        "# `id` is a unique identifier — it carries no signal and would cause leakage if",
        "# accidentally fed into the model, so we drop it. We also drop the empty",
        "# `Unnamed: 32` column commonly present in the Kaggle CSV (guard with `errors='ignore'`).",
        "df = df.drop(columns=['id'])",
        "df = df.drop(columns=['Unnamed: 32'], errors='ignore')",
        "print('Shape after dropping `id`:', df.shape)",
        "df.head()",
    ),
    md("### 1.4 Countplot of `diagnosis` (magma palette)"),
    code(
        "plt.figure(figsize=(6, 4))",
        "sns.countplot(x='diagnosis', data=df, palette='magma')",
        "plt.title('Diagnosis distribution', fontweight='bold')",
        "plt.xlabel('Diagnosis (B = Benign, M = Malignant)')",
        "plt.ylabel('Count')",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # --- 2. Preprocessing ---
    md(
        "## 2. Data Preprocessing, Building Models and Evaluation",
        "",
        "### 2.1 Counts of unique rows in the `diagnosis` column",
    ),
    code(
        "counts = df['diagnosis'].value_counts()",
        "print(counts)",
        "print(f'\\nMalignant share: {counts[\"M\"] / counts.sum() * 100:.1f}%')",
    ),
    md("### 2.2 Map categorical values to numerical values"),
    code(
        "# Convention: malignant = 1 (positive class), benign = 0",
        "df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})",
        "print(df['diagnosis'].value_counts().rename({0: 'Benign (0)', 1: 'Malignant (1)'}))",
    ),
    md("### 2.3 Split into train and test"),
    code(
        "X = df.drop(columns=['diagnosis'])",
        "y = df['diagnosis']",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE",
        ")",
        "",
        "print(f'Train: {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% malignant)')",
        "print(f'Test : {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% malignant)')",
    ),
    md(
        "We use a **stratified** split so train and test keep the same malignant/benign",
        "ratio. KNN, Logistic Regression and SVM are sensitive to feature scale, so each",
        "of those models lives inside a `Pipeline` that runs `StandardScaler` first. The",
        "scaler is fitted **only on the training fold** to avoid leakage. Random Forest",
        "is scale-invariant and gets the raw features.",
    ),

    # --- Helper ---
    md("### 2.4 Helper to score every model uniformly"),
    code(
        "results = []  # one row per model",
        "",
        "def evaluate(name, model, X_te=X_test, y_te=y_test):",
        "    pred = model.predict(X_te)",
        "    acc = accuracy_score(y_te, pred)",
        "    results.append({'Model': name, 'Accuracy': acc})",
        "    print(f'{name} accuracy: {acc:.4f}')",
        "    return acc",
    ),

    # --- 2.5 LR ---
    md("### 2.5 Logistic Regression"),
    code(
        "lr = Pipeline([",
        "    ('scaler', StandardScaler()),",
        "    ('clf', LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),",
        "])",
        "lr.fit(X_train, y_train)",
        "evaluate('Logistic Regression', lr)",
    ),

    # --- 2.6 KNN ---
    md("### 2.6 K-Nearest Neighbors"),
    code(
        "knn = Pipeline([",
        "    ('scaler', StandardScaler()),",
        "    ('clf', KNeighborsClassifier(n_neighbors=5)),",
        "])",
        "knn.fit(X_train, y_train)",
        "evaluate('K-Nearest Neighbors', knn)",
    ),

    # --- 2.7 Random Forest ---
    md("### 2.7 Random Forest"),
    code(
        "rf = RandomForestClassifier(",
        "    n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1",
        ")",
        "rf.fit(X_train, y_train)",
        "evaluate('Random Forest', rf)",
    ),

    # --- 2.8 SVM ---
    md("### 2.8 Support Vector Machine"),
    code(
        "svm = Pipeline([",
        "    ('scaler', StandardScaler()),",
        "    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)),",
        "])",
        "svm.fit(X_train, y_train)",
        "evaluate('SVM', svm)",
    ),

    # --- Comparison ---
    md("## 3. Comparison — which model is the best?"),
    code(
        "results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)",
        "print(results_df.round(4))",
        "",
        "ax = results_df.set_index('Model')['Accuracy'].plot(",
        "    kind='barh', figsize=(9, 4.5), color='steelblue', edgecolor='white',",
        ")",
        "ax.set_xlim(0, 1.02)",
        "ax.set_title('Test-set accuracy by model', fontweight='bold')",
        "for container in ax.containers:",
        "    ax.bar_label(container, fmt='%.4f', padding=3)",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "best = results_df.iloc[0]",
        "print(f\"\\nBest model: {best['Model']} (accuracy = {best['Accuracy']:.4f})\")",
    ),
    code(
        "# Inspect the best model with confusion matrix + full classification report",
        "model_lookup = {",
        "    'Logistic Regression': lr,",
        "    'K-Nearest Neighbors': knn,",
        "    'Random Forest': rf,",
        "    'SVM': svm,",
        "}",
        "best_model = model_lookup[best['Model']]",
        "y_pred = best_model.predict(X_test)",
        "",
        "fig, ax = plt.subplots(figsize=(5, 4))",
        "ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),",
        "                      display_labels=['Benign', 'Malignant']).plot(",
        "    ax=ax, cmap='Blues', colorbar=False,",
        ")",
        "ax.set_title(f\"Confusion matrix — {best['Model']}\", fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))",
    ),

    # --- Interpretation ---
    md(
        "## 4. Interpretation",
        "",
        "**Which model wins?** The bar chart above ranks the four classifiers by test-set",
        "accuracy. On this dataset the gap between the top models is usually within ~2",
        "percentage points, which is **smaller than the variance** caused by the random",
        "split — so 'winning by one point' is not always meaningful. We should look at the",
        "confusion matrix and class-level metrics of the top model to decide.",
        "",
        "**Why all models do well.** The 30 features (mean / standard error / worst of 10",
        "tumour-cell measurements like radius, texture, perimeter, area) are very",
        "informative on their own — the two classes are well separated in this feature",
        "space. Even simple linear classifiers reach ~95%+ accuracy.",
        "",
        "**Class-specific reading.** In a medical context, the **False Negatives** (a",
        "malignant case classified as benign) are the most costly mistakes, because the",
        "patient may not get follow-up care. The confusion matrix above lets us check this",
        "directly. If FN looks too high we should:",
        "1. Lower the decision threshold below 0.5 to trade some precision for more recall",
        "   on the positive (malignant) class.",
        "2. Or use `class_weight='balanced'` in LR/SVM, or `class_weight='balanced'` /",
        "   tune `n_estimators` in RF.",
        "",
        "**Caveats.** The dataset has only 569 patients — a single train/test split has",
        "high variance. For a serious comparison we would average accuracy across",
        "stratified k-fold cross-validation and also report **ROC-AUC** (threshold-",
        "agnostic, more robust to imbalance).",
        "",
        "**Next steps.** Run `GridSearchCV` over each model's main hyperparameters",
        "(`C` for LR/SVM, `n_neighbors` for KNN, `max_depth` and `n_estimators` for RF)",
        "and re-rank — that often closes the gap further and tells us which model has the",
        "most headroom from tuning.",
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

out_path = os.path.join(os.path.dirname(__file__), "dailychallenge.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
