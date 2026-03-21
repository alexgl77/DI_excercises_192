import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# Save plots in the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load Dataset ──────────────────────────────────────────────────────────────
path = r"C:\Users\alexg\Downloads\ds_salary_dataset\Data Science Job Salary dataset\datascience_salaries.csv"
df = pd.read_csv(path, index_col=0)
print("Dataset loaded.")

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ─── Task 1: Min-Max Normalization of Salary ──────────────────────────────────
print("\n=== Task 1: Min-Max Normalization ===")

# Use salary_in_usd for consistent comparison (salary column uses mixed currencies)
salary_col = 'salary'

print(f"Using column: '{salary_col}'")
print(f"Before normalization:")
print(f"  Min:  ${df[salary_col].min():,.0f}")
print(f"  Max:  ${df[salary_col].max():,.0f}")
print(f"  Mean: ${df[salary_col].mean():,.0f}")

# Apply Min-Max scaling: scales all values to range [0, 1]
# Formula: (x - min) / (max - min)
scaler = MinMaxScaler()
df['salary_normalized'] = scaler.fit_transform(df[[salary_col]])

print(f"\nAfter normalization (salary_normalized):")
print(f"  Min:  {df['salary_normalized'].min():.4f}")
print(f"  Max:  {df['salary_normalized'].max():.4f}")
print(f"  Mean: {df['salary_normalized'].mean():.4f}")

# Visualize before vs after
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[salary_col].hist(bins=30, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title(f'Salary - Before Normalization')
axes[0].set_xlabel('USD')

df['salary_normalized'].hist(bins=30, ax=axes[1], color='seagreen', edgecolor='white')
axes[1].set_title('Salary - After Min-Max Normalization')
axes[1].set_xlabel('Normalized Value [0, 1]')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'salary_normalization.png'))
plt.close()
print("Saved: salary_normalization.png")

# ─── Task 2: Dimensionality Reduction with PCA ────────────────────────────────
print("\n=== Task 2: PCA Dimensionality Reduction ===")

# PCA only works on numeric data — encode categorical columns first
df_pca = df.copy()

# Encode all categorical/object columns with LabelEncoder
le = LabelEncoder()
for col in df_pca.select_dtypes(include=['object', 'category']).columns:
    df_pca[col] = le.fit_transform(df_pca[col].astype(str))

# Drop the normalized salary (derived column) to avoid redundancy
# Use all original numeric features
features = [c for c in df_pca.columns if c != 'salary_normalized']
X = df_pca[features].fillna(0)  # fill any residual NaN

print(f"Features going into PCA: {list(X.columns)}")
print(f"Original dimensions: {X.shape[1]} features")

# Scale before PCA (PCA is sensitive to scale)
X_scaled = MinMaxScaler().fit_transform(X)

# Apply PCA — reduce to 2 components for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(f"\nReduced to 2 principal components:")
print(f"  PC1 explains: {explained[0]*100:.1f}% of variance")
print(f"  PC2 explains: {explained[1]*100:.1f}% of variance")
print(f"  Total variance retained: {sum(explained)*100:.1f}%")

# Add PCA components to dataframe for inspection
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Visualize PCA result colored by experience level
exp_col = 'experience_level'
exp_codes = df[exp_col].astype('category').cat.codes

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['PC1'], df['PC2'], c=exp_codes, cmap='tab10', alpha=0.6, s=30)
plt.title('PCA - Data Science Salaries (2 Components)')
plt.xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')

# Add legend with actual experience level labels
exp_labels = df[exp_col].astype('category').cat.categories
handles = [plt.Line2D([0],[0], marker='o', color='w',
           markerfacecolor=plt.cm.tab10(i/len(exp_labels)),
           markersize=8, label=label)
           for i, label in enumerate(exp_labels)]
plt.legend(handles=handles, title='Experience Level')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'pca_result.png'))
plt.close()
print("Saved: pca_result.png")

# ─── Task 3: Group by Experience Level — Avg & Median Salary ─────────────────
print("\n=== Task 3: Salary Aggregation by Experience Level ===")

# Group by experience_level and calculate mean and median
salary_by_exp = df.groupby(exp_col)[salary_col].agg(
    Average='mean',
    Median='median',
    Count='count'
).round(2).sort_values('Average', ascending=False)

print("Salary stats by experience level:")
print(salary_by_exp.to_string())

# Map coded labels to readable names if they are abbreviations
# Typical Kaggle ds_salaries codes: EN=Entry, MI=Mid, SE=Senior, EX=Executive
level_map = {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior', 'EX': 'Executive'}
salary_by_exp.index = [level_map.get(x, x) for x in salary_by_exp.index]

print("\nWith readable labels:")
print(salary_by_exp.to_string())

# Visualize average vs median salary per experience level
x = range(len(salary_by_exp))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar([i - width/2 for i in x], salary_by_exp['Average'], width,
               label='Average', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], salary_by_exp['Median'],  width,
               label='Median',  color='seagreen')

ax.set_xticks(list(x))
ax.set_xticklabels(salary_by_exp.index)
ax.set_ylabel('Salary (USD)')
ax.set_title('Average vs Median Salary by Experience Level')
ax.legend()

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'salary_by_experience.png'))
plt.close()
print("Saved: salary_by_experience.png")

print("\n=== Summary ===")
print(f"Dataset shape:              {df.shape}")
print(f"Salary range:               ${df[salary_col].min():,.0f} – ${df[salary_col].max():,.0f}")
print(f"Normalized salary range:    {df['salary_normalized'].min():.2f} – {df['salary_normalized'].max():.2f}")
print(f"PCA variance retained:      {sum(explained)*100:.1f}%")
print(f"Experience levels analyzed: {list(salary_by_exp.index)}")
