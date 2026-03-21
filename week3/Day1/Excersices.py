import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Exercise 1: Introduction to Data Analysis ────────────────────────────────
"""
ESSAY: What is Data Analysis?

Data analysis is the process of inspecting, cleaning, transforming, and modeling
data to discover useful information, draw conclusions, and support decision-making.

WHY IS IT IMPORTANT?
- Turns raw data into actionable insights for better decisions
- Reduces uncertainty by grounding decisions in evidence
- Reveals hidden patterns and trends
- Drives efficiency by identifying bottlenecks in processes
- Enables personalization at scale

THREE APPLICATION AREAS:

1. Healthcare: Predict disease outbreaks, improve diagnosis accuracy, and personalize
   treatment plans by analyzing patient records and medical history.

2. Finance: Fraud detection, credit scoring, and risk management. Banks analyze
   transaction patterns in real time to flag suspicious activity.

3. Retail/E-Commerce: Recommendation engines, demand forecasting, and customer
   behavior analysis. Amazon and Netflix use purchase/view history to suggest
   products/content customers are likely to engage with.
"""

# ─── Exercise 2: Dataset Loading and Initial Analysis ─────────────────────────

# Dataset 1: Credit Card Approvals (UCI - publicly available)
credit_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
credit_cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16']
try:
    credit_df = pd.read_csv(credit_url, header=None, names=credit_cols, na_values='?')
    print("=== Credit Card Approvals Dataset ===")
    print(f"Shape: {credit_df.shape}")
    print(credit_df.head())
except Exception as e:
    print(f"Credit dataset error: {e}")

# Dataset 2: Iris (via seaborn - same as Kaggle version)
iris = sns.load_dataset('iris')
print("\n=== Iris Dataset ===")
print(f"Shape: {iris.shape}")
print(iris.head())

# Dataset 3: Sleep - load from Kaggle CSV if available, else skip
# sleep_df = pd.read_csv('path/to/sleeping_data.csv')
# print(sleep_df.head())
print("\nNote: Sleep dataset requires manual Kaggle download.")
print("Load with: sleep_df = pd.read_csv('sleeping_data.csv')")

# ─── Exercise 3: Identifying Data Types ───────────────────────────────────────
print("\n=== Exercise 3: Data Type Classification ===")

print("\nCredit Card Approvals columns:")
credit_types = {
    'A1':  ('Qualitative',  'Gender - categorical (b/a)'),
    'A2':  ('Quantitative', 'Age - continuous numeric'),
    'A3':  ('Quantitative', 'Debt - continuous numeric'),
    'A4':  ('Qualitative',  'Marital status - categorical'),
    'A5':  ('Qualitative',  'Bank customer type - categorical'),
    'A6':  ('Qualitative',  'Education level - categorical'),
    'A7':  ('Qualitative',  'Ethnicity - categorical'),
    'A8':  ('Quantitative', 'Years employed - continuous'),
    'A9':  ('Qualitative',  'Prior default - binary'),
    'A10': ('Qualitative',  'Employed - binary'),
    'A11': ('Quantitative', 'Credit score - discrete numeric'),
    'A12': ('Qualitative',  'Driver license - binary'),
    'A13': ('Qualitative',  'Citizen type - categorical'),
    'A14': ('Quantitative', 'Zip code - discrete'),
    'A15': ('Quantitative', 'Income - continuous numeric'),
    'A16': ('Qualitative',  'Approved - target label (+/-)'),
}
for col, (dtype, reason) in credit_types.items():
    print(f"  {col}: {dtype} — {reason}")

# ─── Exercise 4: Exploring Data Types - Iris ──────────────────────────────────
print("\n=== Exercise 4: Iris Data Types ===")
iris_types = {
    'sepal_length': ('Quantitative', 'Continuous - measured in cm'),
    'sepal_width':  ('Quantitative', 'Continuous - measured in cm'),
    'petal_length': ('Quantitative', 'Continuous - measured in cm'),
    'petal_width':  ('Quantitative', 'Continuous - measured in cm'),
    'species':      ('Qualitative',  'Nominal - category with no natural order'),
}
for col, (dtype, reason) in iris_types.items():
    print(f"  {col}: {dtype} — {reason}")

# ─── Exercise 5: Basic Observation Skills ─────────────────────────────────────
print("\n=== Exercise 5: Interesting Columns for Analysis ===")
print("Using Iris dataset as stand-in for sleep dataset demonstration:")
print(iris.describe())
print("""
Interesting columns for analysis:
  - sepal_length / petal_length: Trend analysis across species
  - species: Group comparison (do different species have significantly different measurements?)
  - petal_width: Could reveal natural clusters when compared to petal_length
""")

# ─── Exercise 6: Structured vs Unstructured ───────────────────────────────────
print("\n=== Exercise 6: Structured vs Unstructured Data ===")
exercise6 = [
    ("Company financial reports in Excel",     "Structured",   "Rows/columns with defined data types"),
    ("Photographs on social media",            "Unstructured", "Raw pixel data, no predefined schema"),
    ("Collection of news articles",            "Unstructured", "Free-form text, no consistent fields"),
    ("Inventory data in relational database",  "Structured",   "Relational tables with defined schema"),
    ("Recorded interviews (market research)",  "Unstructured", "Audio waveforms, no predefined format"),
]
for source, dtype, reason in exercise6:
    print(f"  {source}")
    print(f"    → {dtype}: {reason}")

# ─── Exercise 7: Unstructured to Structured Transformation ────────────────────
print("\n=== Exercise 7: Transformation Methods ===")
exercise7 = [
    ("Blog posts about travel",        "NLP + NER → extract: destination, date, activities, sentiment_score"),
    ("Customer service call recordings","Speech-to-Text → extract: duration, issue_category, resolution, sentiment"),
    ("Handwritten brainstorming notes","OCR → categorize into: idea, category, priority, owner"),
    ("Cooking video tutorial",         "Video analysis + transcription → extract: step, ingredient, quantity, duration"),
]
for source, method in exercise7:
    print(f"  {source}")
    print(f"    → {method}")

# ─── Exercise 8: Import Titanic from GitHub ────────────────────────────────────
print("\n=== Exercise 8: Titanic Dataset from GitHub ===")
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    titanic_df = pd.read_csv(titanic_url)
    print(f"Shape: {titanic_df.shape}")
    print(f"Columns: {list(titanic_df.columns)}")
    print(titanic_df.head())
except Exception as e:
    print(f"Error loading Titanic dataset: {e}")

# ─── Exercise 9: Export DataFrame to Excel and JSON ───────────────────────────
print("\n=== Exercise 9: Export to Excel and JSON ===")
employees = pd.DataFrame({
    'Name':       ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age':        [25, 30, 35, 28],
    'Department': ['Marketing', 'Engineering', 'Sales', 'HR'],
    'Salary':     [55000, 85000, 62000, 58000]
})
print(employees)

employees.to_excel('employees.xlsx', index=False)
print("\nExported to employees.xlsx")

employees.to_json('employees.json', orient='records', indent=2)
print("Exported to employees.json")

# Verify
print("\nVerification - reading back employees.json:")
print(pd.read_json('employees.json'))

# ─── Exercise 10: Reading JSON Data from URL ──────────────────────────────────
print("\n=== Exercise 10: JSON Data from URL ===")
json_url = "https://jsonplaceholder.typicode.com/users"
try:
    json_df = pd.read_json(json_url)
    print(f"Shape: {json_df.shape}")
    print(f"Columns: {list(json_df.columns)}")
    print("\nFirst 5 entries:")
    print(json_df.head())
except Exception as e:
    print(f"Error loading JSON: {e}")
