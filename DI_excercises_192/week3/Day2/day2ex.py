import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─── Load Titanic Dataset ──────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Titanic dataset loaded.")
print(f"Shape: {df.shape}")
print(df.head())

# ─── Exercise 1: Duplicate Detection and Removal ──────────────────────────────
print("\n=== Exercise 1: Duplicate Detection and Removal ===")

rows_before = len(df)
print(f"Rows before: {rows_before}")

# Check how many duplicate rows exist
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {n_duplicates}")

# Remove duplicates (keeps first occurrence)
df = df.drop_duplicates()

rows_after = len(df)
print(f"Rows after: {rows_after}")
print(f"Rows removed: {rows_before - rows_after}")
# Titanic is already clean so 0 duplicates is expected

# ─── Exercise 2: Handling Missing Values ──────────────────────────────────────
print("\n=== Exercise 2: Handling Missing Values ===")

print("Missing values per column:")
print(df.isnull().sum())

# --- Age: impute with median (robust to outliers) ---
# Using SimpleImputer from sklearn
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']])
print(f"\nAge missing after imputation: {df['Age'].isnull().sum()}")

# --- Embarked: fill with most frequent value (2 rows missing) ---
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(f"Embarked missing after fill: {df['Embarked'].isnull().sum()}")

# --- Cabin: too many missing (~77%) — drop the column entirely ---
print(f"Cabin missing: {df['Cabin'].isnull().sum()} / {len(df)} rows — dropping column")
df = df.drop(columns=['Cabin'])

print("\nMissing values after treatment:")
print(df.isnull().sum())

# ─── Exercise 3: Feature Engineering ─────────────────────────────────────────
print("\n=== Exercise 3: Feature Engineering ===")

# --- Family Size: SibSp (siblings/spouse) + Parch (parents/children) + self ---
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("FamilySize sample:")
print(df['FamilySize'].value_counts().sort_index())

# --- Title: extract from Name column using regex ---
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
print("\nRaw titles found:")
print(df['Title'].value_counts())

# Consolidate rare titles into broader categories
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
print("\nCleaned titles:")
print(df['Title'].value_counts())

# --- Encode Title with Label Encoding (will be re-done properly in Exercise 6) ---
le = LabelEncoder()
df['Title_encoded'] = le.fit_transform(df['Title'])
print("\nTitle encoded sample:")
print(df[['Title', 'Title_encoded']].drop_duplicates().sort_values('Title_encoded'))

# ─── Exercise 4: Outlier Detection and Handling ───────────────────────────────
print("\n=== Exercise 4: Outlier Detection and Handling ===")

# --- Visualize Fare and Age distributions ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Fare'].plot(kind='box', ax=axes[0], title='Fare - Before')
df['Age'].plot(kind='box', ax=axes[1], title='Age - Before')
plt.tight_layout()
plt.savefig('outliers_before.png')
plt.close()
print("Saved: outliers_before.png")

# --- IQR method to detect outliers ---
def iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

fare_low, fare_high = iqr_bounds(df['Fare'])
age_low,  age_high  = iqr_bounds(df['Age'])
print(f"\nFare IQR bounds: [{fare_low:.2f}, {fare_high:.2f}]")
print(f"Age  IQR bounds: [{age_low:.2f}, {age_high:.2f}]")
print(f"Fare outliers: {((df['Fare'] < fare_low) | (df['Fare'] > fare_high)).sum()}")
print(f"Age  outliers: {((df['Age']  < age_low)  | (df['Age']  > age_high)).sum()}")

# --- Handle Fare: quantile capping at 0.98 (cap extreme fares, keep distribution) ---
fare_cap = df['Fare'].quantile(0.98)
print(f"\nFare cap at 0.98 quantile: {fare_cap:.2f}")
df['Fare'] = df['Fare'].clip(upper=fare_cap)

# --- Handle Fare: also apply log transformation to reduce skew ---
df['Fare_log'] = np.log1p(df['Fare'])  # log1p handles 0 values safely

# --- Age: IQR bounds are reasonable — no extreme capping needed ---
# Age_high is ~73, which is a valid age. No rows removed.
print(f"Age max after imputation: {df['Age'].max():.1f} — within IQR bounds, no treatment needed")

# Compare before/after Fare
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Fare'].plot(kind='box', ax=axes[0], title='Fare - After Capping')
df['Fare_log'].plot(kind='box', ax=axes[1], title='Fare (log) - After Log Transform')
plt.tight_layout()
plt.savefig('outliers_after.png')
plt.close()
print("Saved: outliers_after.png")

# ─── Exercise 5: Data Standardization and Normalization ───────────────────────
print("\n=== Exercise 5: Standardization and Normalization ===")

# StandardScaler: for Age (roughly normal after imputation)
scaler_std = StandardScaler()
df['Age_scaled'] = scaler_std.fit_transform(df[['Age']])
print(f"Age_scaled  — mean: {df['Age_scaled'].mean():.4f}, std: {df['Age_scaled'].std():.4f}")

# MinMaxScaler: for Fare_log (skewed/bounded, log already applied)
scaler_mm = MinMaxScaler()
df['Fare_normalized'] = scaler_mm.fit_transform(df[['Fare_log']])
print(f"Fare_normalized — min: {df['Fare_normalized'].min():.4f}, max: {df['Fare_normalized'].max():.4f}")

# ─── Exercise 6: Feature Encoding ─────────────────────────────────────────────
print("\n=== Exercise 6: Feature Encoding ===")

# --- Sex: binary — label encode (male=1, female=0) ---
df['Sex_encoded'] = (df['Sex'] == 'male').astype(int)
print("Sex encoded (male=1, female=0):")
print(df['Sex_encoded'].value_counts())

# --- Embarked: nominal — one-hot encoding ---
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
print("\nEmbarked one-hot columns added:")
print(df[['Embarked', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].head())

# --- Title: nominal — one-hot encoding (replace label encoded column) ---
df = df.drop(columns=['Title_encoded'])  # remove the label encoded version
title_dummies = pd.get_dummies(df['Title'], prefix='Title')
df = pd.concat([df, title_dummies], axis=1)
print("\nTitle one-hot columns added:")
print([c for c in df.columns if c.startswith('Title_')])

# ─── Exercise 7: Data Transformation for Age Feature ─────────────────────────
print("\n=== Exercise 7: Age Groups ===")

# Create age bins for life stages
age_bins   = [0, 12, 18, 60, 100]
age_labels = ['Child', 'Teen', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
print("Age group distribution:")
print(df['AgeGroup'].value_counts().sort_index())

# One-hot encode the age groups
age_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
df = pd.concat([df, age_dummies], axis=1)
print("\nAgeGroup one-hot columns:")
print(df[['AgeGroup', 'AgeGroup_Child', 'AgeGroup_Teen', 'AgeGroup_Adult', 'AgeGroup_Senior']].head(10))

# ─── Final Summary ─────────────────────────────────────────────────────────────
print("\n=== Final Dataset Summary ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
