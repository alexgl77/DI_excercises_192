"""
Download the DI template, fill the 5 "Your answer" cells with actual answers,
write the filled notebook to disk and emit base64 for upload.
"""

import base64
import json


# The base64 content I fetched from Drive (saved to template_b64.txt)
with open("template_b64.txt", "r", encoding="utf-8") as f:
    b64 = f.read().strip()

raw = base64.b64decode(b64).decode("utf-8")
nb = json.loads(raw)


# Answers keyed by the id of the markdown cell to replace
ANSWERS = {
    # Exercise 1
    "9b165d7a": [
        "### Your answer\n",
        "\n",
        "**Prediction objective.** I will build a binary classification model whose target variable is `default` (1 = the borrower failed to repay within 12 months of origination, 0 = the borrower repaid on time); the model output is the probability `P(default)` used to support the credit team's approve/reject decision and the calibration of interest rates by risk band.\n",
        "\n",
        "**Data types and why each one helps.**\n",
        "- *Personal / demographic data* (age, marital status, dependents, education, housing status) helps because life-stage variables proxy income stability and financial obligations.\n",
        "- *Financial data* (monthly income, fixed expenses, debt-to-income ratio, employment tenure, employment type, declared assets) directly captures the borrower's ability to repay.\n",
        "- *Credit data* (credit score, prior delinquencies, recent inquiries, credit utilization) is the single strongest predictor because it summarizes past repayment behavior.\n",
        "- *Loan data* (requested amount, term, rate, purpose, collateral, co-signers) measures exposure size and the contractual structure of the risk.\n",
        "- *Behavioral data* (internal account usage, salary deposits, overdraft frequency) refines risk for existing customers where we have first-party signals.\n",
        "\n",
        "**Realistic sources and how to integrate them.** I would extract the bank's internal records from the core banking system and CRM through a scheduled ETL into our data warehouse, pull external credit bureau data (Equifax, Experian, TransUnion, Dicom) via their official APIs against the applicant's national ID, complement with external scoring providers under existing contracts, ingest public macroeconomic indicators (regional unemployment, inflation) from the central bank or national statistics office, and where the customer has explicitly consented use open banking / PSD2 feeds to enrich income and cashflow signals.\n",
        "\n",
        "**Risks and constraints.** The project must comply with privacy regulation (GDPR, Chilean law 19.628 on personal data, banking secrecy), document the legal basis for processing each data category, and audit fairness so that the model does not produce disparate impact on protected attributes such as gender, age, or race; we also need to manage data-quality risks (missing values, stale bureau snapshots), sampling bias (we only observe defaults among loans we approved historically), and governance requirements such as model documentation, challenger models, and periodic re-validation by the risk committee.",
    ],

    # Exercise 2
    "ed0e3a74": [
        "### Your justification\n",
        "\n",
        "**Selected features.** I would keep `credit_score`, `debt_to_income`, `annual_income`, `loan_amount`, `interest_rate`, `employment_length`, `num_delinquencies`, `total_utilization`, `num_open_accounts`, `home_ownership`, `purpose`, and `term`.\n",
        "\n",
        "**Justification.**\n",
        "- `credit_score` is the strongest single predictor because it already aggregates years of repayment behavior across institutions.\n",
        "- `debt_to_income` captures the borrower's current leverage and directly measures repayment capacity given the new loan.\n",
        "- `annual_income` and `loan_amount` together define the loan-to-income ratio, which drives stress under adverse scenarios.\n",
        "- `interest_rate` is informative both as a price signal and as a proxy for the lender's prior risk assessment; higher rates correlate with higher default rates.\n",
        "- `employment_length` reflects job stability and the persistence of income.\n",
        "- `num_delinquencies` and `total_utilization` measure recent stress signals that the credit score may not yet have fully absorbed.\n",
        "- `num_open_accounts` indicates credit hunger or recent leverage build-up.\n",
        "- `home_ownership`, `purpose`, and `term` add structural risk information: owning a home is correlated with lower default, certain purposes (debt consolidation, small business) carry higher risk, and longer terms accumulate more uncertainty.\n",
        "\n",
        "**Excluded features and why.**\n",
        "- `state` and `zip_code` are excluded as direct inputs because they correlate strongly with protected attributes and would introduce fairness risk and geographic redlining; instead, I would derive aggregated, non-sensitive features such as regional unemployment rate.\n",
        "- `application_type` is excluded if it is constant or near-constant in the available data, since it provides no discriminative signal.\n",
        "- Any unique identifier (e.g., `loan_id`, `applicant_id`) is excluded because it carries no predictive signal and risks data leakage.\n",
        "\n",
        "**Encoding categorical features.** I would one-hot encode low-cardinality categorical variables such as `home_ownership` and `purpose`, and for higher-cardinality categorical variables I would use target encoding fitted strictly on the training fold to avoid leakage; for ordinal categories such as `term` I would map the levels to integers that preserve the order.\n",
        "\n",
        "**Imputing missing values.** I would impute numeric variables using the median computed on the training set, and for categorical variables I would either use the most frequent category or introduce an explicit `missing` category, always paired with a binary `is_missing` flag so the model can learn whether the missingness itself is informative.",
    ],

    # Exercise 3
    "cdd4054c": [
        "### Your answer\n",
        "\n",
        "**Candidate models.** I would start with a Logistic Regression as a transparent baseline because its coefficients translate directly to log-odds and satisfy the explainability requirements of banking regulators; in parallel I would train a Gradient Boosting model (XGBoost, LightGBM, or CatBoost) which typically delivers state-of-the-art performance on tabular data and captures non-linear interactions among features such as the joint effect of `debt_to_income`, `credit_score`, and `loan_amount`.\n",
        "\n",
        "**Evaluation plan.** I would split the data into a stratified 80/20 train/test partition to preserve the default rate in both subsets, then run stratified 5-fold cross-validation on the training set for hyperparameter tuning, and reserve the test set for a single final evaluation reported at the end. The primary metric is ROC-AUC because it is robust to class imbalance and threshold-agnostic, complemented by PR-AUC (more informative when defaults are rare), recall on the positive class (defaults caught), precision (rejected applicants who were truly risky), the F1 / F-beta score (with beta > 1 if missing a default is more costly than rejecting a good customer), the confusion matrix, and a calibration plot to confirm that predicted probabilities match observed default frequencies. The decision threshold is not fixed at 0.5; instead, I would compute the expected business cost as a function of the threshold using the cost of false negatives (write-off of the loan principal) versus false positives (lost interest margin) and pick the threshold that minimizes expected cost.\n",
        "\n",
        "**Handling class imbalance.** I would use stratification in every split so that minority class proportions are preserved, and within the model I would either set `class_weight='balanced'` (Logistic Regression and tree-based models support this natively) or apply SMOTE oversampling inside a pipeline so that resampling only happens on the training fold of each cross-validation split, never on validation data.\n",
        "\n",
        "**Hyperparameter optimization and leakage prevention.** I would use Bayesian or randomized search over a defined hyperparameter space (regularization strength for Logistic Regression; `max_depth`, `learning_rate`, `n_estimators`, `min_child_samples` for Gradient Boosting), wrapping all preprocessing (imputation, encoding, scaling, resampling) inside a `Pipeline` that is fit only on the training fold; this guarantees that imputation statistics, target-encoding means, and oversampled rows never see validation data, eliminating the most common source of data leakage.",
    ],

    # Exercise 4
    "d27bd365": [
        "### Your answer\n",
        "\n",
        "**1) Predicting stock prices.** This is a supervised learning problem framed as time-series regression. The input is a sequence of historical observations per asset (open, high, low, close, volume, derived technical indicators such as RSI, MACD, moving averages, and optionally macroeconomic features), the output is a continuous estimate of the price (or return) at a future horizon such as `t+1`, `t+5`, or `t+30`, and the learning signal is the supervised pair `(features at time t, observed price at t+h)` from history, which lets the model minimize a regression loss such as MSE or MAE between predicted and realized prices.\n",
        "\n",
        "**2) Organizing a library of books.** This is an unsupervised learning problem framed as clustering (and, if interpretable themes are required, topic modeling). The input is a vector representation of each book derived from its content (TF-IDF over the synopsis or full text, or sentence-transformer embeddings) plus structured metadata (author, publication year), the output is a cluster label assigned to each book so that books with similar content end up in the same group, and the learning signal is the geometry of the data itself (minimizing within-cluster distances and maximizing between-cluster distances), because there are no pre-existing genre labels to supervise the model.\n",
        "\n",
        "**3) Programming a robot to navigate a maze.** This is naturally a reinforcement learning problem (and degenerates to classical search if the maze is fully known and static). The input is the agent's state representation (its current position, surrounding walls, and possibly its sensor readings), the output is an action chosen from a discrete action space (move up, down, left, right), and the learning signal is a scalar reward provided by the environment (positive when reaching the goal, small negative per step to incentivize short paths, large negative for collisions), which the agent maximizes over full episodes by learning a policy via algorithms such as Q-Learning, SARSA, or Deep Q-Networks; if the maze is known and static, a classical search algorithm such as A* or BFS finds the optimal path without any learning at all.",
    ],

    # Exercise 5
    "e7d9f0ef": [
        "### Your answer\n",
        "\n",
        "**Supervised learning — classification (e.g., spam vs non-spam).** I would evaluate this model with a stratified train/validation/test split followed by stratified k-fold cross-validation (k = 5 or 10) so that the class proportions are preserved in every fold. The metrics I would report are accuracy (only when the classes are balanced), precision and recall (treated separately because their relative cost depends on the use case — in spam detection, recall on spam matters but precision on legitimate mail matters even more to avoid lost messages), the F1-score for a balanced summary, ROC-AUC as a threshold-agnostic ranking quality measure, and PR-AUC when the positive class is rare. I would complement these with a confusion matrix and ROC / Precision-Recall curves to choose the operating threshold based on business cost. The key challenges in this category are class imbalance, data leakage caused by preprocessing fit on the full dataset, and concept drift in production that silently degrades the model after deployment.\n",
        "\n",
        "**Unsupervised learning — clustering (e.g., customer segmentation with K-Means).** Without ground-truth labels I would assess the model with indirect (internal) metrics such as the silhouette score (cohesion versus separation), the Davies-Bouldin index (lower is better), and the Calinski-Harabasz index (higher is better), and I would pick the number of clusters using the elbow method on within-cluster sum of squares (WCSS) versus k together with the silhouette score across k. When partial labels are available I would also report the Adjusted Rand Index and Normalized Mutual Information. Beyond numeric metrics, I would inspect cluster centroids and a sample of members per cluster to verify they make business sense, and I would test stability by re-running the clustering on bootstrap subsamples and different random seeds. The main challenges here are that 'good' depends on the downstream objective rather than a single number, K-Means assumes spherical clusters of similar size and breaks on real-world distributions, and high-dimensional data suffers from the curse of dimensionality, which makes distance-based metrics less informative without dimensionality reduction.\n",
        "\n",
        "**Reinforcement learning (e.g., a game-playing agent).** I would evaluate the agent inside a simulated environment by running multiple random seeds and reporting the cumulative reward per episode (mean and standard deviation across seeds) over a fixed evaluation budget, the convergence behavior (how many episodes are needed to stabilize the reward), the sample efficiency (reward achieved per environment interaction), the success rate when the task has a clear success criterion, and the exploration-vs-exploitation balance measured through the epsilon decay schedule or policy entropy. I would also assess generalization by evaluating the trained agent on environments that differ slightly from the training distribution. The key challenges are sparse rewards (the agent receives almost no learning signal during most episodes), high variance across random seeds that makes a single run misleading, overfitting to the training environment so that the policy fails in new settings, and the substantial computational cost of running millions of environment interactions for non-trivial problems.",
    ],
}


# Apply
replaced = 0
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid in ANSWERS:
        cell["source"] = ANSWERS[cid]
        replaced += 1

print(f"Replaced {replaced} answer cells")

# Save filled notebook
with open("exercise_filled.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Emit base64 for upload
with open("exercise_filled.ipynb", "rb") as f:
    b64_out = base64.b64encode(f.read()).decode("ascii")
with open("filled_b64.txt", "w") as f:
    f.write(b64_out)

print(f"Filled notebook: exercise_filled.ipynb")
print(f"Base64 length: {len(b64_out)}")
