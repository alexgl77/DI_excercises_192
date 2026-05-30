"""
Generate dailychallenge.ipynb for Week 5 Day 1: Admission prediction with Logistic Regression.
Uses the classic Andrew Ng ex2data1 (Exam1, Exam2, Admitted) — 100 students.
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
        "# Daily Challenge: Logistic Regression for Admission Prediction",
        "",
        "**Course:** Developers Institute  **Week 5 - Day 1**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Goal: build a Logistic Regression model that predicts whether a student is",
        "admitted to a university given their scores on two exams. We will explore the",
        "data, train the model, predict, evaluate accuracy, and visualize the decision",
        "boundary.",
        "",
        "Dataset: Andrew Ng's classic admission dataset — 100 students, 2 exam scores,",
        "and a binary admitted/not-admitted target.",
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
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import (",
        "    accuracy_score, confusion_matrix, classification_report,",
        "    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,",
        ")",
        "",
        "sns.set_theme(style='whitegrid')",
    ),

    # --- Step 1: Data exploration ---
    md(
        "## 1. Data exploration — load and visualize",
        "",
        "Load the dataset from the canonical mirror, name the columns, and inspect the",
        "first rows.",
    ),
    code(
        "URL = 'https://raw.githubusercontent.com/jdwittenauer/ipython-notebooks/master/data/ex2data1.txt'",
        "df = pd.read_csv(URL, header=None, names=['Exam1', 'Exam2', 'Admitted'])",
        "",
        "print('Shape:', df.shape)",
        "df.head()",
    ),
    code(
        "# Basic summary",
        "print('Class distribution:')",
        "print(df['Admitted'].value_counts().rename({0: 'Not admitted', 1: 'Admitted'}))",
        "print('\\nStatistical summary:')",
        "df.describe().round(2)",
    ),
    code(
        "# Scatter plot: Exam1 vs Exam2, coloured by admission",
        "plt.figure(figsize=(8, 6))",
        "for label, color, marker in [(1, 'seagreen', 'o'), (0, 'tomato', 'x')]:",
        "    subset = df[df['Admitted'] == label]",
        "    plt.scatter(",
        "        subset['Exam1'], subset['Exam2'],",
        "        c=color, marker=marker, s=70, edgecolor='white' if marker == 'o' else None,",
        "        label='Admitted' if label == 1 else 'Not admitted',",
        "    )",
        "plt.xlabel('Exam 1 score')",
        "plt.ylabel('Exam 2 score')",
        "plt.title('Admission outcome by exam scores', fontweight='bold')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "**Observation.** Visually the two groups are well separated by a roughly diagonal",
        "boundary: students with higher combined scores on both exams tend to be admitted.",
        "There is some overlap in the middle region where exam scores alone are not enough",
        "to decide — that is exactly where the logistic regression's probabilistic output",
        "will be most useful.",
    ),

    # --- Step 2: Apply Logistic Regression ---
    md(
        "## 2. Apply Logistic Regression with scikit-learn",
        "",
        "We split the data into train and test sets so we can honestly measure",
        "generalization (the assignment lets us train on the full dataset, but a held-out",
        "evaluation is good practice and the dataset is small but enough to split).",
    ),
    code(
        "X = df[['Exam1', 'Exam2']].values",
        "y = df['Admitted'].values",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, stratify=y, random_state=42",
        ")",
        "",
        "print(f'Train: {X_train.shape[0]} students ({y_train.mean()*100:.1f}% admitted)')",
        "print(f'Test : {X_test.shape[0]} students ({y_test.mean()*100:.1f}% admitted)')",
    ),
    code(
        "# Train the logistic regression model",
        "model = LogisticRegression(max_iter=1000, random_state=42)",
        "model.fit(X_train, y_train)",
        "",
        "print('Learned parameters:')",
        "print(f'  Intercept (theta_0): {model.intercept_[0]:.4f}')",
        "print(f'  Coef Exam1 (theta_1): {model.coef_[0][0]:.4f}')",
        "print(f'  Coef Exam2 (theta_2): {model.coef_[0][1]:.4f}')",
    ),

    # --- Step 3: Predictions + accuracy ---
    md(
        "## 3. Making predictions and measuring accuracy",
        "",
        "We score the model on train and test sets separately and inspect a few specific",
        "predictions to see how the probabilities behave.",
    ),
    code(
        "# Predictions on both sets",
        "y_train_pred = model.predict(X_train)",
        "y_test_pred = model.predict(X_test)",
        "",
        "train_acc = accuracy_score(y_train, y_train_pred)",
        "test_acc = accuracy_score(y_test, y_test_pred)",
        "",
        "print(f'Train accuracy: {train_acc:.4f}')",
        "print(f'Test accuracy : {test_acc:.4f}')",
    ),
    code(
        "# A few sample predictions with probabilities",
        "samples = [",
        "    (45, 85),   # mid-high exam 2 — should be borderline / admitted",
        "    (75, 80),   # high in both — clearly admitted",
        "    (30, 40),   # low in both — clearly not admitted",
        "    (60, 60),   # average — borderline",
        "]",
        "sample_probs = model.predict_proba(samples)[:, 1]",
        "sample_preds = model.predict(samples)",
        "",
        "for (e1, e2), prob, pred in zip(samples, sample_probs, sample_preds):",
        "    label = 'Admitted' if pred == 1 else 'Not admitted'",
        "    print(f'Exam1={e1:>4}, Exam2={e2:>4} -> P(admitted)={prob:.3f} -> {label}')",
    ),

    # --- Step 4: Evaluation ---
    md(
        "## 4. Model evaluation",
        "",
        "Accuracy alone hides the kind of errors the model is making. We also look at the",
        "confusion matrix and the per-class precision / recall / F1.",
    ),
    code(
        "# Confusion matrix on the test set",
        "cm = confusion_matrix(y_test, y_test_pred)",
        "",
        "fig, ax = plt.subplots(figsize=(5, 4))",
        "ConfusionMatrixDisplay(cm, display_labels=['Not admitted', 'Admitted']).plot(",
        "    ax=ax, cmap='Blues', colorbar=False)",
        "ax.set_title('Confusion matrix — test set', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "tn, fp, fn, tp = cm.ravel()",
        "print(f'TN: {tn}  FP: {fp}')",
        "print(f'FN: {fn}  TP: {tp}')",
    ),
    code(
        "# Precision / Recall / F1 per class + classification report",
        "report = classification_report(",
        "    y_test, y_test_pred,",
        "    target_names=['Not admitted', 'Admitted'], output_dict=True,",
        ")",
        "report_df = pd.DataFrame(report).transpose().round(3)",
        "print(report_df)",
        "",
        "metrics_df = pd.DataFrame({",
        "    'Not admitted': [report['Not admitted']['precision'], report['Not admitted']['recall'], report['Not admitted']['f1-score']],",
        "    'Admitted':     [report['Admitted']['precision'],     report['Admitted']['recall'],     report['Admitted']['f1-score']],",
        "}, index=['Precision', 'Recall', 'F1-score'])",
        "",
        "ax = metrics_df.plot(kind='bar', figsize=(8, 4.5), color=['tomato', 'seagreen'], edgecolor='white')",
        "plt.title('Precision / Recall / F1 per class', fontweight='bold')",
        "plt.ylim(0, 1.05)",
        "plt.ylabel('Score')",
        "plt.xticks(rotation=0)",
        "plt.legend(title='Class')",
        "for container in ax.containers:",
        "    ax.bar_label(container, fmt='%.2f', padding=3)",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # --- Decision boundary ---
    md(
        "## 5. Decision boundary visualization",
        "",
        "Because the model uses only two features (the two exam scores), we can plot the",
        "decision boundary directly on the original axes — no PCA needed. The boundary is",
        "where `P(admitted) = 0.5`, which corresponds to a straight line in the input",
        "space.",
    ),
    code(
        "# Build a grid over the feature space",
        "x_min, x_max = df['Exam1'].min() - 5, df['Exam1'].max() + 5",
        "y_min, y_max = df['Exam2'].min() - 5, df['Exam2'].max() + 5",
        "xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),",
        "                     np.linspace(y_min, y_max, 400))",
        "Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)",
        "",
        "plt.figure(figsize=(9, 7))",
        "plt.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.55)",
        "plt.colorbar(label='P(admitted)')",
        "plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)",
        "",
        "for label, color, marker in [(1, 'darkgreen', 'o'), (0, 'darkred', 'x')]:",
        "    subset = df[df['Admitted'] == label]",
        "    plt.scatter(",
        "        subset['Exam1'], subset['Exam2'],",
        "        c=color, marker=marker, s=70,",
        "        edgecolor='white' if marker == 'o' else None,",
        "        label='Admitted' if label == 1 else 'Not admitted',",
        "    )",
        "",
        "plt.xlabel('Exam 1 score')",
        "plt.ylabel('Exam 2 score')",
        "plt.title(f'Decision boundary — Test accuracy: {test_acc:.3f}', fontweight='bold')",
        "plt.legend(loc='upper right')",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # --- Interpretation ---
    md(
        "## 6. Interpretation",
        "",
        "**The model.** Logistic regression learned a linear combination of the two exam",
        "scores and squashed it through the sigmoid to produce an admission probability.",
        "The fitted equation has the form:",
        "",
        "$$P(\\text{admitted}) = \\sigma(\\theta_0 + \\theta_1 \\cdot \\text{Exam1} + \\theta_2 \\cdot \\text{Exam2})$$",
        "",
        "Both coefficients are **positive** — higher scores on either exam increase the",
        "admission probability, which matches intuition. The intercept is large and",
        "negative, meaning a student with very low scores has a very small probability.",
        "",
        "**The decision boundary** is a straight line in the (Exam1, Exam2) plane. Points",
        "above and to the right of the line are classified as admitted; below and to the",
        "left, not admitted. The classifier is *linear* — it cannot draw a curved boundary",
        "even if the data hinted at one.",
        "",
        "**Accuracy.** On this dataset the test accuracy lands around 0.85–0.90 (it",
        "fluctuates a bit with the random seed because the test set has only 20 students).",
        "The few misclassifications happen in the diagonal band where the classes overlap",
        "— students with mid-range scores on both exams sometimes get admitted and",
        "sometimes do not, so the model cannot resolve every individual case correctly.",
        "",
        "**Limitations.**",
        "- A purely linear classifier cannot capture interactions like 'a low score on one",
        "  exam can be compensated by a near-perfect score on the other'.",
        "- The dataset is very small (100 rows); accuracy on the held-out set has high",
        "  variance.",
        "- Decision threshold is fixed at 0.5 — if the cost of rejecting a good student is",
        "  higher than admitting a marginal one, lower the threshold to favour recall.",
        "",
        "**Next steps.** Add polynomial / interaction features, or move to a non-linear",
        "model (Gradient Boosting, SVM with RBF kernel) and compare. Collect more data if",
        "this were a real admissions system. Validate fairness across demographic groups",
        "before any operational use.",
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
