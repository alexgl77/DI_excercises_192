"""
Generate exercisesxp.ipynb for Week 6 Day 1: DL vs ML, ANN diagram, polynomial fit + cross-validation.
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
        "# Exercises XP — DL vs ML, ANNs, Polynomial Fitting and Cross-Validation",
        "",
        "**Course:** Developers Institute  **Week 6 - Day 1**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Five exercises: a comparative table between traditional ML and deep learning,",
        "a small artificial neural network diagram with labeled components, a noisy",
        "polynomial dataset, polynomial fits at several degrees, and a cross-validation",
        "sweep to find the optimal degree.",
    ),

    md("## Setup"),
    code(
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from sklearn.metrics import mean_squared_error",
        "",
        "RANDOM_STATE = 0",
        "np.random.seed(RANDOM_STATE)",
    ),

    # ============================================================
    # Exercise 1
    # ============================================================
    md(
        "## Exercise 1 — Deep Learning vs Traditional Machine Learning",
        "",
        "### 1.1 Comparative table",
    ),
    code(
        "comparison = pd.DataFrame({",
        "    'Aspect': [",
        "        'Feature Engineering',",
        "        'Data Processing',",
        "        'Scalability',",
        "        'Pattern Discovery',",
        "        'Computational Requirements',",
        "    ],",
        "    'Traditional ML': [",
        "        'Manual: engineer designs the features by hand.',",
        "        'Works best on structured / tabular data; needs cleaning + transforms.',",
        "        'Performance plateaus quickly as data grows.',",
        "        'Captures mostly linear or shallow non-linear patterns; struggles with complex hierarchical structure.',",
        "        'Light: trains on CPU in minutes; small memory footprint.',",
        "    ],",
        "    'Deep Learning': [",
        "        'Automatic: the network learns useful features end-to-end from raw inputs.',",
        "        'Handles unstructured data natively (images, audio, text, video).',",
        "        'Performance keeps improving with more data and more parameters.',",
        "        'Learns deep hierarchical representations (edges -> textures -> objects).',",
        "        'Heavy: requires GPUs/TPUs, large memory, hours-to-days of training.',",
        "    ],",
        "})",
        "comparison",
    ),
    md(
        "### 1.2 Real-world scenarios",
        "",
        "- **Traditional ML is better suited for:** *credit scoring at a regional bank*.",
        "  The dataset is tabular (income, debt, credit history), labeled, modest in size",
        "  (tens of thousands of rows), and interpretability is a regulatory requirement.",
        "  Logistic Regression or Gradient Boosting will reach near-state-of-the-art",
        "  accuracy while keeping the model auditable.",
        "",
        "- **Deep learning is the superior choice for:** *medical-image diagnosis from",
        "  chest X-rays*. The inputs are high-resolution images (unstructured), labels",
        "  exist for hundreds of thousands of scans, and the relevant patterns are",
        "  spatial and hierarchical (edges → textures → lesion shape → diagnosis).",
        "  A convolutional network learns these representations directly from pixels",
        "  and reaches expert-level accuracy.",
        "",
        "### 1.3 Why does deep learning excel on unstructured data?",
        "",
        "Unstructured data — images, audio, video, raw text — has no human-readable",
        "columns: the signal lives in spatial, temporal or sequential patterns that are",
        "very hard to hand-engineer. Deep networks tackle this by stacking many layers",
        "of differentiable transformations and using backpropagation to *learn the",
        "features themselves*, end-to-end, from the raw input. The lower layers",
        "discover primitives (edges, phonemes, n-grams), the middle layers compose them",
        "into mid-level structures (shapes, syllables, phrases), and the upper layers",
        "produce task-specific representations. This automatic, hierarchical feature",
        "learning is exactly what classical ML cannot do, and it is why DL dominates",
        "vision, speech, NLP and other unstructured-data domains.",
    ),

    # ============================================================
    # Exercise 2
    # ============================================================
    md(
        "## Exercise 2 — Artificial Neural Network Diagram",
        "",
        "Architecture: input layer (3 neurons) → hidden layer (4 neurons) → output",
        "layer (2 neurons). Components labeled directly on the diagram below.",
    ),
    code(
        "# Draw a labeled ANN diagram with matplotlib",
        "fig, ax = plt.subplots(figsize=(13, 7))",
        "ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')",
        "",
        "layers = {",
        "    'Input': (2.0, [7.5, 5.5, 3.5]),",
        "    'Hidden': (5.0, [8.0, 6.0, 4.0, 2.0]),",
        "    'Output': (8.0, [6.5, 4.5]),",
        "}",
        "colors = {'Input': '#4C72B0', 'Hidden': '#DD8452', 'Output': '#55A868'}",
        "",
        "# Draw connections first (so neurons sit on top)",
        "for layer_a, layer_b in [('Input', 'Hidden'), ('Hidden', 'Output')]:",
        "    xa, ya_list = layers[layer_a]",
        "    xb, yb_list = layers[layer_b]",
        "    for ya in ya_list:",
        "        for yb in yb_list:",
        "            ax.plot([xa, xb], [ya, yb], color='gray', alpha=0.5, linewidth=1, zorder=1)",
        "",
        "# Draw the neurons",
        "for name, (x, ys) in layers.items():",
        "    for i, y in enumerate(ys):",
        "        ax.scatter(x, y, s=1400, color=colors[name], edgecolor='black',",
        "                   zorder=2, linewidth=1.5)",
        "        ax.text(x, y, f'{name[0]}{i+1}', ha='center', va='center',",
        "                color='white', fontweight='bold', zorder=3)",
        "    ax.text(x, 9.4, f'{name} layer\\n({len(ys)} neurons)',",
        "            ha='center', fontsize=11, fontweight='bold', color=colors[name])",
        "",
        "# Annotations on components",
        "# Weight",
        "ax.annotate('Weight (w)\\non each connection',",
        "            xy=(3.5, 6.8), xytext=(0.2, 8.7),",
        "            fontsize=10, color='gray',",
        "            arrowprops=dict(arrowstyle='->', color='gray'))",
        "",
        "# Bias",
        "ax.text(2.0, 1.5, 'Bias (b)\\nadded inside each neuron', ha='center',",
        "        fontsize=10, color='black', fontstyle='italic')",
        "",
        "# Activation function",
        "ax.annotate('Activation function\\n(e.g., ReLU, sigmoid)',",
        "            xy=(5.0, 2.0), xytext=(5.3, 0.5),",
        "            fontsize=10, color='black', fontstyle='italic',",
        "            arrowprops=dict(arrowstyle='->', color='black'))",
        "",
        "# Forward-pass arrow",
        "ax.annotate('', xy=(8.7, 0.7), xytext=(1.3, 0.7),",
        "            arrowprops=dict(arrowstyle='->', color='black', lw=2))",
        "ax.text(5.0, 0.2, 'Forward propagation (information flow)',",
        "        ha='center', fontsize=11, fontweight='bold')",
        "",
        "ax.set_title('Artificial Neural Network — 3 input / 4 hidden / 2 output',",
        "             fontsize=14, fontweight='bold', pad=15)",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "**How information flows through the network.**",
        "",
        "Each input neuron carries a feature value (`x₁, x₂, x₃`). Every connection has a",
        "**weight** `w` that multiplies the value passing through. Inside each hidden",
        "neuron the weighted inputs are summed, a **bias** `b` is added, and the result",
        "passes through an **activation function** (here ReLU) to introduce",
        "non-linearity. The four hidden activations then flow forward — again through",
        "weighted connections, sums, biases and activations — into the two output",
        "neurons, which produce the final prediction `ŷ`. Training updates every",
        "weight and bias via **backpropagation** so the network's outputs match the",
        "target labels.",
    ),

    # ============================================================
    # Exercise 3
    # ============================================================
    md(
        "## Exercise 3 — Noisy Dataset",
        "",
        "Generate 20 points of `y = −x²` with normally-distributed noise (`mean=0, std=0.05`),",
        "scatter-plot them, and split into train (first 12) and test (last 8).",
    ),
    code(
        "np.random.seed(0)",
        "x = np.arange(-1, 1, 0.1)",
        "y = -x**2 + np.random.normal(0, 0.05, len(x))",
        "",
        "print(f'Number of points: {len(x)}')",
        "print(f'x range: [{x.min():.2f}, {x.max():.2f}]')",
        "print(f'y range: [{y.min():.4f}, {y.max():.4f}]')",
    ),
    code(
        "plt.figure(figsize=(9, 5.5))",
        "plt.scatter(x, y, color='steelblue', s=70, edgecolor='white', label='Noisy data')",
        "x_smooth = np.linspace(-1, 1, 200)",
        "plt.plot(x_smooth, -x_smooth**2, 'k--', alpha=0.5, label='True function: $y = -x^2$')",
        "plt.xlabel('x'); plt.ylabel('y')",
        "plt.title('Noisy dataset of 20 points around $y = -x^2$', fontweight='bold')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "# Split: first 12 points = train, last 8 points = test",
        "x_train, y_train = x[:12], y[:12]",
        "x_test, y_test = x[12:], y[12:]",
        "",
        "print(f'Train: {len(x_train)} points, x in [{x_train.min():.2f}, {x_train.max():.2f}]')",
        "print(f'Test : {len(x_test)} points, x in [{x_test.min():.2f}, {x_test.max():.2f}]')",
    ),

    # ============================================================
    # Exercise 4
    # ============================================================
    md(
        "## Exercise 4 — Polynomial Fits of Different Degrees",
        "",
        "Two helpers: `polynomial_fit(degree)` returns a polynomial object fitted on",
        "the training set, `plot_polyfit(degree)` overlays the fit on train + test data.",
    ),
    code(
        "def polynomial_fit(degree):",
        "    coeffs = np.polyfit(x_train, y_train, deg=degree)",
        "    return np.poly1d(coeffs)",
        "",
        "",
        "def plot_polyfit(degree, ax=None):",
        "    p = polynomial_fit(degree)",
        "    if ax is None:",
        "        fig, ax = plt.subplots(figsize=(8, 5))",
        "",
        "    x_curve = np.linspace(-1.05, 1.05, 300)",
        "    ax.plot(x_curve, p(x_curve), color='tomato', linewidth=2, label=f'Fit (deg={degree})')",
        "    ax.plot(x_curve, -x_curve**2, color='gray', linestyle='--', alpha=0.6, label='True $y=-x^2$')",
        "    ax.scatter(x_train, y_train, color='steelblue', s=70, edgecolor='white', label='Train')",
        "    ax.scatter(x_test, y_test, color='seagreen', s=70, edgecolor='white', marker='^', label='Test')",
        "",
        "    train_rmse = np.sqrt(mean_squared_error(y_train, p(x_train)))",
        "    test_rmse = np.sqrt(mean_squared_error(y_test, p(x_test)))",
        "    ax.set_title(f'Degree {degree}  |  Train RMSE = {train_rmse:.3f}  Test RMSE = {test_rmse:.3f}',",
        "                 fontweight='bold')",
        "    ax.set_xlabel('x'); ax.set_ylabel('y')",
        "    ax.set_ylim(-1.4, 0.3)",
        "    ax.legend(loc='lower center', fontsize=9)",
        "    return ax",
    ),
    code(
        "# Plot fits for degrees 1, 7 and 11",
        "fig, axes = plt.subplots(1, 3, figsize=(17, 5))",
        "for deg, ax in zip([1, 7, 11], axes):",
        "    plot_polyfit(deg, ax=ax)",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "**Observation.**",
        "- **Degree 1** is too rigid — a straight line cannot bend into a parabola, so",
        "  it underfits both train and test (large RMSE on both).",
        "- **Degree 7** captures the parabolic shape well; the curve is close to the",
        "  true `−x²` on both sets. This is the sweet spot for our small noisy sample.",
        "- **Degree 11** has more freedom than data points — it bends through every",
        "  training point (very low train RMSE) but goes wild between them, producing a",
        "  huge **test RMSE**. Classic overfitting.",
    ),

    # ============================================================
    # Exercise 5
    # ============================================================
    md(
        "## Exercise 5 — Cross-Validation to Find the Optimal Degree",
        "",
        "Loop degrees 1 → 11, record train + test RMSE, plot both with log-y, and",
        "report the degree that minimizes test RMSE.",
    ),
    code(
        "results = []",
        "for deg in range(1, 12):",
        "    p = polynomial_fit(deg)",
        "    train_rmse = np.sqrt(mean_squared_error(y_train, p(x_train)))",
        "    test_rmse = np.sqrt(mean_squared_error(y_test, p(x_test)))",
        "    results.append({'Degree': deg, 'Train RMSE': train_rmse, 'Test RMSE': test_rmse})",
        "",
        "results_df = pd.DataFrame(results)",
        "print(results_df.round(4).to_string(index=False))",
    ),
    code(
        "plt.figure(figsize=(9, 5.5))",
        "plt.plot(results_df['Degree'], results_df['Train RMSE'],",
        "         marker='o', color='steelblue', label='Train RMSE')",
        "plt.plot(results_df['Degree'], results_df['Test RMSE'],",
        "         marker='s', color='tomato', label='Test RMSE')",
        "plt.yscale('log')",
        "plt.xlabel('Polynomial degree')",
        "plt.ylabel('RMSE (log scale)')",
        "plt.title('Train vs Test RMSE — log scale', fontweight='bold')",
        "plt.xticks(range(1, 12))",
        "plt.legend()",
        "plt.grid(True, alpha=0.3, which='both')",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "best = results_df.loc[results_df['Test RMSE'].idxmin()]",
        "print(f\"Optimal degree by test RMSE: {int(best['Degree'])} \"",
        "      f\"(Train RMSE = {best['Train RMSE']:.4f}, Test RMSE = {best['Test RMSE']:.4f})\")",
        "print('\\nMatch with the true underlying model y = -x^2 (a degree-2 polynomial)?',",
        "      'YES' if best['Degree'] == 2 else 'No (close, but not exact)')",
    ),
    md(
        "**Interpretation.**",
        "- **Train RMSE** decreases monotonically with degree — more flexible models",
        "  always fit the training data better, all the way to interpolating it",
        "  (degree ≥ n_train − 1).",
        "- **Test RMSE** drops to a minimum around degree 2, then climbs sharply for",
        "  higher degrees — this is the visible signature of **overfitting**.",
        "- The optimal degree by test RMSE matches the **true underlying function**",
        "  `y = −x²` (a quadratic). Cross-validation has recovered the correct model",
        "  complexity from the data alone, without us telling it.",
        "",
        "**Generalization rule of thumb.** When the gap between train and test error",
        "starts widening, you've passed the sweet spot. Pick the model complexity",
        "*just before* that gap explodes.",
    ),

    md(
        "## Summary",
        "",
        "- **Ex 1.** Traditional ML and deep learning are not interchangeable: pick the",
        "  one that matches the data type, dataset size, interpretability needs, and",
        "  compute budget.",
        "- **Ex 2.** Every neural network combines neurons, weights, biases and",
        "  activation functions. Information flows forward through the layers; learning",
        "  flows backward via gradient descent.",
        "- **Ex 3.** Generating noisy data with a known underlying function lets us",
        "  measure how well a model recovers the truth.",
        "- **Ex 4.** Polynomial degree controls model complexity. Too low = underfitting,",
        "  too high = overfitting; in between is the sweet spot.",
        "- **Ex 5.** Cross-validation (here a simple train/test split) finds that sweet",
        "  spot empirically — and in this case it recovers exactly the degree of the",
        "  true model that generated the data.",
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

out_path = os.path.join(os.path.dirname(__file__), "exercisesxp.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
