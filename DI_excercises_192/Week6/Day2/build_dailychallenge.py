"""
Generate dailychallenge.ipynb for Week 6 Day 2:
Classifying handwritten digits with FCNN vs CNN.
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
        "# Daily Challenge: MNIST — Fully-Connected vs Convolutional Networks",
        "",
        "**Course:** Developers Institute  **Week 6 - Day 2**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Train two models on MNIST and compare them honestly: a **Fully-Connected**",
        "network (dense layers only, image flattened to 784 pixels) and a",
        "**Convolutional Neural Network** that exploits the 2D spatial structure of",
        "the digits. We use the same optimizer, loss, batch size and epoch budget so",
        "the comparison is fair.",
    ),

    md("## Setup"),
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
        "import seaborn as sns",
        "",
        "import tensorflow as tf",
        "from tensorflow import keras",
        "from tensorflow.keras import layers, models",
        "from tensorflow.keras.datasets import mnist",
        "from tensorflow.keras.utils import to_categorical",
        "",
        "sns.set_theme(style='whitegrid')",
        "RANDOM_STATE = 42",
        "tf.random.set_seed(RANDOM_STATE)",
        "np.random.seed(RANDOM_STATE)",
        "",
        "print('TF version:', tf.__version__)",
    ),

    # ============================================================
    # 1. Load MNIST
    # ============================================================
    md("## 1. Load the MNIST Dataset"),
    code(
        "(x_train, y_train), (x_test, y_test) = mnist.load_data()",
        "print(f'x_train shape: {x_train.shape}  | y_train shape: {y_train.shape}')",
        "print(f'x_test shape : {x_test.shape}  | y_test shape : {y_test.shape}')",
        "print(f'Pixel range  : {x_train.min()} - {x_train.max()}')",
    ),
    code(
        "# Quick look at a few examples",
        "fig, axes = plt.subplots(2, 5, figsize=(11, 5))",
        "for digit, ax in zip(range(10), axes.flat):",
        "    idx = np.where(y_train == digit)[0][0]",
        "    ax.imshow(x_train[idx], cmap='gray')",
        "    ax.set_title(f'Label: {digit}')",
        "    ax.axis('off')",
        "plt.suptitle('MNIST — one sample per class', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # ============================================================
    # 2. Preprocess for the FCNN
    # ============================================================
    md(
        "## 2. Preprocess the Data for the Fully-Connected Network",
        "",
        "Flatten each 28×28 image to a 784-vector, normalize pixels to `[0, 1]`,",
        "one-hot encode the labels.",
    ),
    code(
        "# Flatten + normalize",
        "x_train_fc = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0",
        "x_test_fc = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0",
        "",
        "# One-hot encode the labels",
        "y_train_cat = to_categorical(y_train, num_classes=10)",
        "y_test_cat = to_categorical(y_test, num_classes=10)",
        "",
        "print(f'x_train_fc shape: {x_train_fc.shape}')",
        "print(f'y_train_cat shape: {y_train_cat.shape}')",
        "print(f'One-hot example : {y_train[0]} -> {y_train_cat[0]}')",
    ),

    # ============================================================
    # 3. Build + train the FCNN
    # ============================================================
    md(
        "## 3. Build and Train the Fully-Connected Network",
        "",
        "Two hidden layers (256 and 128 ReLU units) + softmax output. Compiled with",
        "Adam + categorical cross-entropy + accuracy.",
    ),
    code(
        "fcnn = models.Sequential([",
        "    layers.Dense(256, activation='relu', input_shape=(784,)),",
        "    layers.Dense(128, activation='relu'),",
        "    layers.Dense(10, activation='softmax'),",
        "])",
        "",
        "fcnn.compile(",
        "    optimizer='adam',",
        "    loss='categorical_crossentropy',",
        "    metrics=['accuracy'],",
        ")",
        "fcnn.summary()",
    ),
    code(
        "history_fcnn = fcnn.fit(",
        "    x_train_fc, y_train_cat,",
        "    epochs=10,",
        "    batch_size=128,",
        "    validation_split=0.1,",
        "    verbose=1,",
        ")",
        "",
        "fc_test_loss, fc_test_acc = fcnn.evaluate(x_test_fc, y_test_cat, verbose=0)",
        "print(f'\\nFCNN — test loss: {fc_test_loss:.4f}  |  test accuracy: {fc_test_acc:.4f}')",
    ),

    # ============================================================
    # 4. Preprocess for the CNN
    # ============================================================
    md(
        "## 4. Preprocess the Data for the CNN",
        "",
        "Reshape the input to `(N, 28, 28, 1)` so it fits a `Conv2D` layer (height,",
        "width, channels). Normalize and one-hot encode just like before.",
    ),
    code(
        "x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0",
        "x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0",
        "",
        "print(f'x_train_cnn shape: {x_train_cnn.shape}')",
        "print(f'x_test_cnn shape : {x_test_cnn.shape}')",
    ),

    # ============================================================
    # 5. Build + train the CNN
    # ============================================================
    md(
        "## 5. Build and Train the CNN",
        "",
        "Two `Conv2D + MaxPool2D` blocks (32 → 64 filters), then Flatten + Dense(128)",
        "+ softmax. Same optimizer/loss/epochs as the FCNN so the comparison is fair.",
    ),
    code(
        "cnn = models.Sequential([",
        "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),",
        "    layers.MaxPool2D((2, 2)),",
        "    layers.Conv2D(64, (3, 3), activation='relu'),",
        "    layers.MaxPool2D((2, 2)),",
        "    layers.Flatten(),",
        "    layers.Dense(128, activation='relu'),",
        "    layers.Dense(10, activation='softmax'),",
        "])",
        "",
        "cnn.compile(",
        "    optimizer='adam',",
        "    loss='categorical_crossentropy',",
        "    metrics=['accuracy'],",
        ")",
        "cnn.summary()",
    ),
    code(
        "history_cnn = cnn.fit(",
        "    x_train_cnn, y_train_cat,",
        "    epochs=10,",
        "    batch_size=128,",
        "    validation_split=0.1,",
        "    verbose=1,",
        ")",
        "",
        "cnn_test_loss, cnn_test_acc = cnn.evaluate(x_test_cnn, y_test_cat, verbose=0)",
        "print(f'\\nCNN — test loss: {cnn_test_loss:.4f}  |  test accuracy: {cnn_test_acc:.4f}')",
    ),

    # ============================================================
    # 6. Compare performance
    # ============================================================
    md("## 6. Compare Performance"),
    code(
        "# Side-by-side accuracy and loss curves",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))",
        "",
        "axes[0].plot(history_fcnn.history['val_accuracy'], marker='o', label='FCNN')",
        "axes[0].plot(history_cnn.history['val_accuracy'], marker='s', label='CNN')",
        "axes[0].set_title('Validation accuracy per epoch', fontweight='bold')",
        "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend()",
        "",
        "axes[1].plot(history_fcnn.history['val_loss'], marker='o', label='FCNN')",
        "axes[1].plot(history_cnn.history['val_loss'], marker='s', label='CNN')",
        "axes[1].set_title('Validation loss per epoch', fontweight='bold')",
        "axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Categorical CE'); axes[1].legend()",
        "",
        "plt.tight_layout()",
        "plt.show()",
    ),
    code(
        "# Summary table",
        "summary = pd.DataFrame({",
        "    'Model': ['FCNN (Dense only)', 'CNN'],",
        "    'Params': [fcnn.count_params(), cnn.count_params()],",
        "    'Test loss': [fc_test_loss, cnn_test_loss],",
        "    'Test accuracy': [fc_test_acc, cnn_test_acc],",
        "})",
        "print(summary.round(4).to_string(index=False))",
        "",
        "# Bar chart of test accuracy",
        "plt.figure(figsize=(8, 4))",
        "colors = ['steelblue', 'seagreen']",
        "bars = plt.bar(summary['Model'], summary['Test accuracy'], color=colors, edgecolor='white')",
        "plt.ylim(0.9, 1.0)",
        "plt.title('Test accuracy — FCNN vs CNN', fontweight='bold')",
        "plt.ylabel('Test accuracy')",
        "for bar, v in zip(bars, summary['Test accuracy']):",
        "    plt.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.4f}',",
        "             ha='center', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "delta = (cnn_test_acc - fc_test_acc) * 100",
        "print(f'CNN beats FCNN by {delta:.2f} percentage points of test accuracy.')",
    ),
    code(
        "# Confusion matrices",
        "from sklearn.metrics import confusion_matrix",
        "",
        "fc_pred = fcnn.predict(x_test_fc, verbose=0).argmax(axis=1)",
        "cnn_pred = cnn.predict(x_test_cnn, verbose=0).argmax(axis=1)",
        "",
        "cm_fc = confusion_matrix(y_test, fc_pred)",
        "cm_cnn = confusion_matrix(y_test, cnn_pred)",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
        "for ax, cm, title in zip(axes, [cm_fc, cm_cnn], ['FCNN', 'CNN']):",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',",
        "                xticklabels=range(10), yticklabels=range(10), ax=ax)",
        "    ax.set_title(f'{title} confusion matrix', fontweight='bold')",
        "    ax.set_xlabel('Predicted'); ax.set_ylabel('True')",
        "plt.tight_layout()",
        "plt.show()",
    ),

    # ============================================================
    # 7. Discussion
    # ============================================================
    md(
        "## 7. Discussion",
        "",
        "**Why does the CNN win?**",
        "",
        "- A Dense layer treats the 784 pixel values as **independent features** —",
        "  it cannot use the 2D structure of the image. Two adjacent pixels are no",
        "  more related to each other than two arbitrary pixels.",
        "- A `Conv2D` layer **scans local 3×3 patches**. It learns features like edges,",
        "  curves and corners that are characteristic of digit strokes — and it reuses",
        "  the same filter weights across the whole image (translation invariance).",
        "- `MaxPool2D` then aggregates these features at coarser scales, so the network",
        "  builds a small hierarchy: edges → strokes → digits.",
        "",
        "**Why the difference is modest on MNIST.**",
        "",
        "MNIST is small and clean (28×28 grayscale, centered digits). A Dense model",
        "with enough capacity can already reach ~98% accuracy. The CNN typically adds",
        "another ~1 percentage point and converges to a lower loss. The gap widens",
        "dramatically on harder datasets (CIFAR-10, real-world images) where the",
        "structural prior of convolutions becomes essential.",
        "",
        "**Parameter efficiency.**",
        "",
        "Look at the parameter counts: the CNN typically has **fewer parameters than",
        "the FCNN** for this task while reaching higher accuracy — because convolutional",
        "weights are shared across the image. That is the real win: better accuracy",
        "with less compute and less risk of overfitting.",
        "",
        "**Takeaway.** Whenever the input has spatial (images), temporal (audio) or",
        "sequential (text) structure, pick an architecture that exploits it. Fully",
        "connected networks are general but inefficient on these data types.",
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
