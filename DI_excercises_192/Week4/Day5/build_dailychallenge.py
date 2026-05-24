"""
Generate dailychallenge.ipynb for Week 4 Day 5.
Topic: Understanding the Essence of Machine Learning.
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
        "# Daily Challenge: Understanding the Essence of Machine Learning",
        "",
        "**Course:** Developers Institute - Data / ML Bootcamp  ",
        "**Author:** Alex Goldbaum  ",
        "**Week 4 - Day 5**",
        "",
        "This notebook explains the core ideas of Machine Learning, the three main paradigms",
        "(Supervised, Unsupervised, Reinforcement), and the typical model-development process,",
        "with a visual flowchart at the end.",
    ),

    # --- Part 1: What is ML and why it matters ---
    md(
        "## 1. What is Machine Learning and why it matters for data analysts",
        "",
        "Machine Learning (ML) is the branch of artificial intelligence in which a computer",
        "learns patterns from data rather than from explicitly written rules. Instead of a",
        "developer hard-coding every condition, the analyst exposes an algorithm to examples",
        "and the algorithm builds a mathematical function that maps inputs to outputs and",
        "generalizes to new, unseen inputs.",
        "",
        "**Why it is important for data analysts:**",
        "",
        "- ML scales pattern discovery beyond what manual analysis can do — it can find",
        "  non-obvious interactions among dozens of variables that a human analyst would",
        "  miss with pivot tables or simple regressions.",
        "- It moves the analytics function from *descriptive* (what happened) and",
        "  *diagnostic* (why it happened) into *predictive* (what will happen) and",
        "  *prescriptive* (what should we do about it).",
        "- It automates repeatable decisions: instead of producing a one-off report, the",
        "  analyst delivers a deployable model that scores new customers, transactions, or",
        "  events continuously in production.",
        "- It forces analytical rigor — defining the target variable, splitting train/test,",
        "  measuring error honestly — which raises the overall quality of data work in the",
        "  organization.",
    ),

    # --- Part 2: Industry applications ---
    md(
        "## 2. Applications across industries (three concrete examples)",
        "",
        "**Finance — credit risk scoring.** Banks train classification models on borrower",
        "history, income, debt, and credit-bureau data to estimate the probability of default",
        "on a new loan. The model output drives approval/rejection decisions, the interest",
        "rate offered, and provisioning for expected losses; the same techniques are used",
        "for fraud detection in card transactions.",
        "",
        "**Healthcare — medical imaging diagnostics.** Convolutional neural networks trained",
        "on labelled X-rays, CT scans, or histopathology slides detect tumors, fractures, or",
        "diabetic retinopathy with accuracy comparable to specialized radiologists. These",
        "models are deployed as triage tools that flag urgent cases first and reduce the",
        "diagnostic backlog.",
        "",
        "**Retail and e-commerce — recommendation systems.** Companies like Amazon, Netflix,",
        "or Spotify use collaborative filtering and deep learning models on user-item",
        "interactions to predict which products, movies, or songs a user will engage with,",
        "directly driving conversion rate, average order value, and retention.",
        "",
        "*Other notable examples:* predictive maintenance in manufacturing (vibration sensors",
        "predicting equipment failure), demand forecasting in logistics, churn prediction in",
        "telecom, autonomous driving in automotive, and personalized pricing in travel.",
    ),

    # --- Part 3: ML types ---
    md(
        "## 3. Supervised vs Unsupervised vs Reinforcement Learning",
        "",
        "### A) Supervised Learning",
        "**Definition.** The dataset is labelled — every training example contains both the",
        "input features and the correct answer (the target). The model learns a function",
        "that maps inputs to outputs by minimizing a loss between its predictions and the",
        "known labels.",
        "",
        "**Example scenario.** Email spam detection: each historical email is labelled as",
        "*spam* or *not spam*, and the model learns to classify new incoming emails.",
        "Other examples: house price prediction (regression), medical diagnosis from",
        "patient data, loan default classification.",
        "",
        "---",
        "",
        "### B) Unsupervised Learning",
        "**Definition.** The dataset has no labels — only the input features. The model",
        "discovers structure or patterns in the data on its own, typically by grouping",
        "similar items, reducing dimensionality, or estimating the underlying distribution.",
        "",
        "**Example scenario.** Customer segmentation: a retailer applies K-Means to",
        "purchase histories and discovers naturally occurring customer groups (e.g.,",
        "*price-sensitive*, *brand loyal*, *occasional buyer*) without anyone labelling",
        "them in advance. Other examples: anomaly detection in network traffic, topic",
        "modeling on text corpora, PCA for visualization.",
        "",
        "---",
        "",
        "### C) Reinforcement Learning (RL)",
        "**Definition.** An agent interacts with an environment by taking actions and",
        "receiving rewards. The agent's objective is to learn a policy — a function from",
        "states to actions — that maximizes the cumulative reward over time, trading off",
        "exploration of new actions against exploitation of known good actions.",
        "",
        "**Example scenario.** AlphaGo and AlphaZero learned to play Go and Chess at",
        "superhuman level by playing millions of games against themselves and updating",
        "their policies based on win/loss rewards. Other examples: robot navigation",
        "and manipulation, dynamic pricing, recommendation engines that adapt online,",
        "autonomous driving control loops.",
    ),

    # --- Part 4: ML development process ---
    md(
        "## 4. The Machine Learning model development process",
        "",
        "Building an ML model is a pipeline. The three central stages — Feature Selection,",
        "Model Selection, and Model Evaluation — sit between problem framing at the start",
        "and deployment + monitoring at the end.",
        "",
        "### 4.1 Feature Selection",
        "Feature selection is the process of choosing which input variables actually go",
        "into the model. This stage matters because irrelevant or redundant features",
        "increase noise, slow training, and hurt generalization, while informative features",
        "improve accuracy and interpretability. Typical techniques include:",
        "- **Filter methods** based on statistics (correlation with the target, mutual",
        "  information, chi-square).",
        "- **Wrapper methods** that fit a model on different feature subsets (recursive",
        "  feature elimination, forward / backward selection).",
        "- **Embedded methods** that perform selection during training (L1-regularized",
        "  models like LASSO, tree-based feature importance, SHAP values).",
        "- **Domain-driven selection** that uses business knowledge to discard features",
        "  with no causal link to the target, or to drop sensitive attributes that create",
        "  fairness or legal risk.",
        "",
        "### 4.2 Model Selection",
        "Model selection is the process of choosing the model family and tuning its",
        "hyperparameters. It depends on the task (classification, regression, clustering,",
        "ranking), the size and dimensionality of the data, the need for explainability,",
        "and the constraints in production (latency, memory). Typical workflow:",
        "- Start with a simple, interpretable baseline (Logistic Regression / Linear",
        "  Regression / Decision Tree).",
        "- Add stronger models (Random Forest, Gradient Boosting, Neural Networks) only",
        "  if the baseline is not enough.",
        "- Tune hyperparameters with cross-validated search (grid, random, or Bayesian).",
        "- Compare candidates fairly on the same validation protocol — never on the test",
        "  set, which is reserved for the final report.",
        "",
        "### 4.3 Model Evaluation",
        "Model evaluation measures whether the trained model meets the business and",
        "technical requirements before deployment. Key principles:",
        "- Hold out a test set that the model never sees during training or tuning.",
        "- Choose metrics aligned with the business cost of mistakes — accuracy is rarely",
        "  enough; use ROC-AUC, PR-AUC, precision/recall, F-beta for classification, and",
        "  MAE / RMSE / MAPE for regression.",
        "- Validate calibration (predicted probabilities should match observed frequencies).",
        "- Stress-test the model on subgroups to detect fairness issues.",
        "- Plan for production monitoring of data drift and metric degradation, since a",
        "  model that performed well at launch can silently decay over weeks or months.",
    ),

    # --- Part 5: Flowchart ---
    md(
        "## 5. ML model development flowchart",
        "",
        "The cell below renders a simple flowchart that visualizes the full pipeline from",
        "problem framing through deployment.",
    ),
    code(
        "import matplotlib.pyplot as plt",
        "from matplotlib.patches import FancyBboxPatch",
        "",
        "steps = [",
        "    ('1. Problem framing',          '#4C72B0'),",
        "    ('2. Data collection',          '#4C72B0'),",
        "    ('3. Data cleaning / EDA',      '#4C72B0'),",
        "    ('4. Feature selection',        '#DD8452'),",
        "    ('5. Model selection + tuning', '#DD8452'),",
        "    ('6. Model evaluation',         '#DD8452'),",
        "    ('7. Deployment',               '#55A868'),",
        "    ('8. Monitoring (drift, AUC)',  '#55A868'),",
        "]",
        "",
        "fig, ax = plt.subplots(figsize=(7, 11))",
        "ax.set_xlim(0, 10)",
        "ax.set_ylim(0, len(steps) * 1.4 + 1)",
        "ax.axis('off')",
        "",
        "y = len(steps) * 1.4",
        "for label, color in steps:",
        "    box = FancyBboxPatch((1, y), 8, 0.9,",
        "                         boxstyle='round,pad=0.1',",
        "                         linewidth=1.2, edgecolor='black',",
        "                         facecolor=color, alpha=0.85)",
        "    ax.add_patch(box)",
        "    ax.text(5, y + 0.45, label, ha='center', va='center',",
        "            color='white', fontsize=11, fontweight='bold')",
        "    if y > 0.7:",
        "        ax.annotate('', xy=(5, y - 0.35), xytext=(5, y - 0.05),",
        "                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))",
        "    y -= 1.4",
        "",
        "ax.set_title('Machine Learning model development pipeline',",
        "             fontsize=13, fontweight='bold', pad=15)",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print('Color legend:')",
        "print('  Blue   -> Data preparation phase')",
        "print('  Orange -> Modeling phase (Feature Selection, Model Selection, Evaluation)')",
        "print('  Green  -> Production phase')",
    ),

    md(
        "## 6. Summary",
        "",
        "Machine Learning lets data analysts move from descriptive reporting to predictive",
        "and prescriptive systems that scale. The three paradigms — Supervised,",
        "Unsupervised, and Reinforcement Learning — differ in whether the data is",
        "labelled, structured by the model itself, or generated through interaction with",
        "an environment, and each fits a different family of business problems. Regardless",
        "of the paradigm, the development pipeline follows the same logic: frame the",
        "problem, prepare the data, **select features**, **select the model**,",
        "**evaluate honestly**, and then deploy with monitoring so that the model keeps",
        "delivering value over time.",
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
