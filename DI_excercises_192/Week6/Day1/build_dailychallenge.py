"""
Generate dailychallenge.ipynb for Week 6 Day 1:
Classification with Neural Networks in TensorFlow (make_circles dataset).
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
        "# Daily Challenge: Classification with Neural Networks in TensorFlow",
        "",
        "**Course:** Developers Institute  **Week 6 - Day 1**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Build, improve, evaluate and visualize neural-network classifiers on the",
        "`make_circles` dataset. We start from a trivial 1-layer model, then add depth,",
        "non-linear activations and a proper train/test split, and finally compare the",
        "decision boundaries learned by each variant.",
    ),

    # ============================================================
    # 1. Classification types
    # ============================================================
    md(
        "## 1. Types of Classification",
        "",
        "**Binary classification** — the target has **two mutually exclusive classes**",
        "(positive / negative, spam / ham). Each example gets exactly one of the two",
        "labels. *Example:* predicting whether an email is spam or not.",
        "",
        "**Multi-class classification** — the target has **more than two mutually",
        "exclusive classes** and each example belongs to exactly one of them. *Example:*",
        "classifying a handwritten digit as 0, 1, 2, … or 9 in MNIST.",
        "",
        "**Multi-label classification** — each example can belong to **several classes",
        "at once**: the labels are not mutually exclusive. *Example:* tagging a news",
        "article with the labels {politics, economy, technology} — an article about a",
        "new tech-policy law can carry all three labels simultaneously.",
        "",
        "The choice of **output layer activation** and **loss function** depends on the",
        "type: sigmoid + binary_crossentropy for binary, softmax + categorical_crossentropy",
        "for multi-class, and sigmoid-per-output + binary_crossentropy for multi-label.",
    ),

    # ============================================================
    # 2. Setup + dataset
    # ============================================================
    md("## 2. Setup and Dataset"),
    code(
        "%pip install -qU tensorflow",
    ),
    code(
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from matplotlib.colors import ListedColormap",
        "",
        "import tensorflow as tf",
        "from tensorflow.keras.models import Sequential",
        "from tensorflow.keras.layers import Dense",
        "from tensorflow.keras.optimizers import SGD, Adam",
        "",
        "from sklearn.datasets import make_circles",
        "from sklearn.model_selection import train_test_split",
        "",
        "RANDOM_STATE = 42",
        "np.random.seed(RANDOM_STATE)",
        "tf.random.set_seed(RANDOM_STATE)",
    ),
    code(
        "samples = 1000",
        "X, y = make_circles(samples, noise=0.03, random_state=42)",
        "",
        "print('X shape:', X.shape, '| y shape:', y.shape)",
        "print('Class balance:', dict(zip(*np.unique(y, return_counts=True))))",
        "print('X sample:', X[:3])",
        "print('y sample:', y[:10])",
    ),
    code(
        "# Visualize the data distribution",
        "plt.figure(figsize=(7, 6))",
        "plt.scatter(X[y == 0, 0], X[y == 0, 1], color='steelblue', s=25, edgecolor='white',",
        "            alpha=0.8, label='Class 0 (outer)')",
        "plt.scatter(X[y == 1, 0], X[y == 1, 1], color='tomato', s=25, edgecolor='white',",
        "            alpha=0.8, label='Class 1 (inner)')",
        "plt.xlabel('Feature 1'); plt.ylabel('Feature 2')",
        "plt.title('make_circles — two concentric clusters', fontweight='bold')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "**Observation.** The two classes form concentric rings — there is no straight",
        "line that can separate them. A linear classifier (or a 1-layer network without",
        "non-linear activations) will fail; we need either non-linear activations or",
        "extra features (e.g., `x² + y²`) to solve this.",
    ),

    # ============================================================
    # 3. Basic model (one dense layer)
    # ============================================================
    md(
        "## 3. Basic Neural Network — One Dense Layer",
        "",
        "Single linear unit, SGD optimizer, no hidden non-linearity. We expect this to",
        "fail on a non-linearly separable problem like `make_circles`.",
    ),
    code(
        "model_basic = Sequential([",
        "    Dense(1, input_shape=(2,)),  # one linear output unit",
        "])",
        "model_basic.compile(",
        "    optimizer=SGD(),",
        "    loss='binary_crossentropy',",
        "    metrics=['accuracy'],",
        ")",
        "model_basic.summary()",
        "",
        "history_basic = model_basic.fit(X, y, epochs=20, verbose=0)",
        "loss_b, acc_b = model_basic.evaluate(X, y, verbose=0)",
        "print(f'Basic model — final loss: {loss_b:.4f} | accuracy: {acc_b:.4f}')",
    ),
    md(
        "**Result.** Accuracy hovers around 50% — the model is at chance level. A",
        "single linear unit cannot separate concentric circles.",
    ),

    # ============================================================
    # 4. Improved model — depth + neurons + Adam
    # ============================================================
    md(
        "## 4. Improve the Model — More Layers, More Neurons, Adam",
        "",
        "Add two hidden layers with ReLU activations, more neurons, and switch the",
        "optimizer from SGD to Adam. The non-linearities give the network the capacity",
        "to bend its decision boundary into a closed shape.",
    ),
    code(
        "tf.random.set_seed(RANDOM_STATE)",
        "model_better = Sequential([",
        "    Dense(16, activation='relu', input_shape=(2,)),",
        "    Dense(8, activation='relu'),",
        "    Dense(1, activation='sigmoid'),",
        "])",
        "model_better.compile(",
        "    optimizer=Adam(learning_rate=0.01),",
        "    loss='binary_crossentropy',",
        "    metrics=['accuracy'],",
        ")",
        "model_better.summary()",
        "",
        "history_better = model_better.fit(X, y, epochs=100, verbose=0)",
        "loss_i, acc_i = model_better.evaluate(X, y, verbose=0)",
        "print(f'Improved model — final loss: {loss_i:.4f} | accuracy: {acc_i:.4f}')",
    ),
    code(
        "# Training curves of the improved model",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))",
        "axes[0].plot(history_better.history['accuracy'], color='steelblue')",
        "axes[0].set_title('Accuracy across epochs', fontweight='bold')",
        "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')",
        "axes[1].plot(history_better.history['loss'], color='tomato')",
        "axes[1].set_title('Loss across epochs', fontweight='bold')",
        "axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Binary cross-entropy')",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # ============================================================
    # 5. Decision boundary visualization
    # ============================================================
    md(
        "## 5. Visualizing the Decision Boundary",
        "",
        "Helper that builds a fine grid over the input space, asks the model to predict",
        "every grid point, and overlays the boundary on the scatter of data points.",
    ),
    code(
        "def plot_decision_boundary(model, X, y, ax=None, title=None):",
        "    if ax is None:",
        "        fig, ax = plt.subplots(figsize=(7, 6))",
        "",
        "    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1",
        "    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1",
        "    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),",
        "                         np.linspace(y_min, y_max, 200))",
        "    grid = np.c_[xx.ravel(), yy.ravel()]",
        "    Z = model.predict(grid, verbose=0)",
        "    if Z.shape[-1] == 1:",
        "        Z = Z.reshape(xx.shape)",
        "    else:",
        "        Z = Z.argmax(axis=1).reshape(xx.shape)",
        "",
        "    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.55)",
        "    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)",
        "    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='steelblue', s=20,",
        "               edgecolor='white', label='Class 0')",
        "    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='tomato', s=20,",
        "               edgecolor='white', label='Class 1')",
        "    if title:",
        "        ax.set_title(title, fontweight='bold')",
        "    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')",
        "    ax.legend(loc='upper right', fontsize=9)",
        "    return ax",
    ),
    code(
        "# Side-by-side: basic vs improved",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
        "plot_decision_boundary(model_basic, X, y, ax=axes[0],",
        "                       title=f'Basic (1 linear unit) — acc {acc_b:.3f}')",
        "plot_decision_boundary(model_better, X, y, ax=axes[1],",
        "                       title=f'Improved (16-8-1 + ReLU + Adam) — acc {acc_i:.3f}')",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "**Observation.** The basic model carves the plane with a single straight line —",
        "useless for concentric rings. The improved model bends the boundary into a",
        "ring-shaped region that wraps around the inner class, closely matching the",
        "true generating geometry.",
    ),

    # ============================================================
    # 6. Activation functions experiment
    # ============================================================
    md(
        "## 6. Activation Functions — ReLU vs Sigmoid (in hidden layers)",
        "",
        "Same architecture, only the hidden activation changes. ReLU is the modern",
        "default — fast to compute and resistant to vanishing gradients. Sigmoid in",
        "hidden layers tends to saturate and slow down learning.",
    ),
    code(
        "def build_model(hidden_activation, optimizer):",
        "    tf.random.set_seed(RANDOM_STATE)",
        "    m = Sequential([",
        "        Dense(16, activation=hidden_activation, input_shape=(2,)),",
        "        Dense(8, activation=hidden_activation),",
        "        Dense(1, activation='sigmoid'),",
        "    ])",
        "    m.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])",
        "    return m",
        "",
        "",
        "model_relu = build_model('relu', Adam(learning_rate=0.01))",
        "model_sigm = build_model('sigmoid', Adam(learning_rate=0.01))",
        "",
        "history_relu = model_relu.fit(X, y, epochs=100, verbose=0)",
        "history_sigm = model_sigm.fit(X, y, epochs=100, verbose=0)",
        "",
        "_, acc_relu = model_relu.evaluate(X, y, verbose=0)",
        "_, acc_sigm = model_sigm.evaluate(X, y, verbose=0)",
        "print(f'ReLU hidden    -> accuracy: {acc_relu:.4f}')",
        "print(f'Sigmoid hidden -> accuracy: {acc_sigm:.4f}')",
    ),
    code(
        "# Compare decision boundaries",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
        "plot_decision_boundary(model_relu, X, y, ax=axes[0],",
        "                       title=f'Hidden = ReLU — acc {acc_relu:.3f}')",
        "plot_decision_boundary(model_sigm, X, y, ax=axes[1],",
        "                       title=f'Hidden = Sigmoid — acc {acc_sigm:.3f}')",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # ============================================================
    # 7. Train/test split + final model
    # ============================================================
    md(
        "## 7. Train/Test Split and Final Model",
        "",
        "80/20 stratified split. We retrain the improved (ReLU + Adam) architecture on",
        "the training set only and evaluate honestly on the held-out test set.",
    ),
    code(
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)",
        "",
        "print(f'Train: {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% class 1)')",
        "print(f'Test : {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% class 1)')",
    ),
    code(
        "tf.random.set_seed(RANDOM_STATE)",
        "model_final = Sequential([",
        "    Dense(32, activation='relu', input_shape=(2,)),",
        "    Dense(16, activation='relu'),",
        "    Dense(8, activation='relu'),",
        "    Dense(1, activation='sigmoid'),",
        "])",
        "model_final.compile(",
        "    optimizer=Adam(learning_rate=0.01),",
        "    loss='binary_crossentropy',",
        "    metrics=['accuracy'],",
        ")",
        "history_final = model_final.fit(",
        "    X_train, y_train, epochs=200,",
        "    validation_data=(X_test, y_test),",
        "    verbose=0,",
        ")",
        "",
        "train_loss, train_acc = model_final.evaluate(X_train, y_train, verbose=0)",
        "test_loss, test_acc = model_final.evaluate(X_test, y_test, verbose=0)",
        "print(f'Train -> loss: {train_loss:.4f} | accuracy: {train_acc:.4f}')",
        "print(f'Test  -> loss: {test_loss:.4f} | accuracy: {test_acc:.4f}')",
    ),

    # ============================================================
    # 8. Evaluate + visualize final
    # ============================================================
    md("## 8. Evaluate and Visualize the Final Model"),
    code(
        "# Training curves with validation",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))",
        "axes[0].plot(history_final.history['accuracy'], label='train')",
        "axes[0].plot(history_final.history['val_accuracy'], label='test')",
        "axes[0].set_title('Final model — accuracy', fontweight='bold')",
        "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend()",
        "",
        "axes[1].plot(history_final.history['loss'], label='train')",
        "axes[1].plot(history_final.history['val_loss'], label='test')",
        "axes[1].set_title('Final model — loss', fontweight='bold')",
        "axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Binary cross-entropy'); axes[1].legend()",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "# Decision boundary on train and test sets",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
        "plot_decision_boundary(model_final, X_train, y_train, ax=axes[0],",
        "                       title=f'Final model — TRAIN (acc {train_acc:.3f})')",
        "plot_decision_boundary(model_final, X_test, y_test, ax=axes[1],",
        "                       title=f'Final model — TEST (acc {test_acc:.3f})')",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "# Confusion matrix on the test set for completeness",
        "from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay",
        "",
        "y_pred = (model_final.predict(X_test, verbose=0) > 0.5).astype(int).ravel()",
        "cm = confusion_matrix(y_test, y_pred)",
        "",
        "fig, ax = plt.subplots(figsize=(5, 4))",
        "ConfusionMatrixDisplay(cm, display_labels=['Class 0', 'Class 1']).plot(",
        "    ax=ax, cmap='Blues', colorbar=False)",
        "ax.set_title('Confusion matrix — test set', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))",
    ),

    # ============================================================
    # 9. Key takeaways
    # ============================================================
    md(
        "## 9. Key Takeaways",
        "",
        "- **Match the model to the geometry of the data.** A linear classifier cannot",
        "  separate concentric circles, no matter how long you train it. Adding non-linear",
        "  hidden activations (`ReLU`) gives the network the capacity to bend its decision",
        "  boundary into the right shape.",
        "",
        "- **Depth + width matter, but a little goes a long way.** Just two hidden layers",
        "  with 16 and 8 neurons take accuracy from ~50% (random) to ~99% on this dataset.",
        "",
        "- **Optimizer matters.** Adam converges much faster and more reliably than vanilla",
        "  SGD here. With the same architecture, Adam reaches near-perfect accuracy in",
        "  about 100 epochs.",
        "",
        "- **Activation function inside the hidden layers matters.** ReLU is the safe",
        "  default: faster training, no saturation. Sigmoid hidden layers learn the same",
        "  problem but more slowly and need more epochs.",
        "",
        "- **Always evaluate on held-out data.** A model that 'looks great' on the training",
        "  set is meaningless until you measure it on data it has not seen. The 80/20",
        "  split + `validation_data` lets us catch overfitting as it happens.",
        "",
        "- **Visualizing the decision boundary** is the fastest way to *see* what the model",
        "  has actually learned. It turns abstract accuracy numbers into a concrete picture",
        "  and is invaluable when debugging classification problems.",
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
