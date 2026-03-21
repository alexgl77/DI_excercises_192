import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET    = r"C:\Users\alexg\Downloads\student_mental_health\Student Mental health.csv"

# Load dataset (used in exercises 4, 5, 6)
df = pd.read_csv(DATASET)

# ---- Exercise 1: Understanding Data Visualization ---------------------------
"""
WHY IS DATA VISUALIZATION IMPORTANT?
Data visualization transforms raw numbers into graphical representations,
making patterns, trends, and outliers immediately visible to the human eye.
Without visualization, identifying relationships in large datasets would require
reading through thousands of rows — a slow and error-prone process. Visuals
accelerate decision-making, improve communication of findings to non-technical
stakeholders, and reveal insights that pure statistics can miss.

PURPOSE OF A LINE GRAPH:
A line graph is used to display data points connected in sequence, making it
ideal for showing change over time (time-series data). The slope and direction
of the line communicate whether a value is rising, falling, or staying flat,
and how quickly the change occurs. Common uses: stock prices, temperature
trends, website traffic over months, etc.
"""
print("Exercise 1: See docstring above for written answers.")

# ---- Exercise 2: Line Plot - Temperature Variation --------------------------
print("\n=== Exercise 2: Temperature Line Plot ===")

days         = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
temperatures = [72, 74, 76, 80, 82, 78, 75]

plt.figure(figsize=(8, 4))
plt.plot(days, temperatures, marker='o', color='steelblue',
         linewidth=2, markersize=7)
plt.fill_between(days, temperatures, alpha=0.1, color='steelblue')
plt.xlabel("Day")
plt.ylabel("Temperature (F)")
plt.title("Temperature Variation Over a Week")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'ex2_temperature.png'))
plt.close()
print("Saved: ex2_temperature.png")

# ---- Exercise 3: Bar Chart - Monthly Sales ----------------------------------
print("\n=== Exercise 3: Monthly Sales Bar Chart ===")

months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales  = [5000, 5500, 6200, 7000, 7500]

plt.figure(figsize=(8, 4))
bars = plt.bar(months, sales, color='seagreen', edgecolor='white', width=0.6)
plt.xlabel("Month")
plt.ylabel("Sales Amount ($)")
plt.title("Monthly Sales - Retail Store")
# Add value labels on top of each bar
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 80,
             f"${bar.get_height():,}",
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'ex3_monthly_sales.png'))
plt.close()
print("Saved: ex3_monthly_sales.png")

# ---- Exercise 4: Histogram - Distribution of CGPA --------------------------
print("\n=== Exercise 4: CGPA Histogram ===")

print(f"Columns: {list(df.columns)}")
print(df['What is your CGPA?'].value_counts())

plt.figure(figsize=(9, 5))
sns.histplot(data=df, x='What is your CGPA?',
             color='slateblue', edgecolor='white', shrink=0.8)
plt.title("Distribution of Students' CGPA")
plt.xlabel("CGPA Range")
plt.ylabel("Number of Students")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'ex4_cgpa_distribution.png'))
plt.close()
print("Saved: ex4_cgpa_distribution.png")

# ---- Exercise 5: Bar Plot - Anxiety Levels Across Genders ------------------
print("\n=== Exercise 5: Anxiety by Gender Bar Plot ===")

# Count anxiety proportions per gender
anxiety_counts = (df.groupby(['Choose your gender', 'Do you have Anxiety?'])
                    .size()
                    .reset_index(name='Count'))

plt.figure(figsize=(8, 5))
sns.barplot(data=anxiety_counts,
            x='Choose your gender',
            y='Count',
            hue='Do you have Anxiety?',
            palette='Set2')
plt.title("Anxiety Levels Across Genders")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.legend(title="Has Anxiety?")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'ex5_anxiety_by_gender.png'))
plt.close()
print("Saved: ex5_anxiety_by_gender.png")

# ---- Exercise 6: Scatter Plot - Age vs Panic Attacks ------------------------
print("\n=== Exercise 6: Age vs Panic Attacks Scatter Plot ===")

# Convert 'Yes'/'No' to numeric 1/0
df['Panic_numeric'] = df['Do you have Panic attack?'].map({'Yes': 1, 'No': 0})

# Add manual jitter to y-axis so points don't all stack on 0 or 1
df['Panic_jitter'] = df['Panic_numeric'] + np.random.uniform(-0.05, 0.05, size=len(df))

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df,
                x='Age',
                y='Panic_jitter',
                hue='Choose your gender',
                palette='Set1',
                alpha=0.7,
                s=80)
plt.title("Relationship Between Age and Panic Attacks")
plt.xlabel("Age")
plt.ylabel("Panic Attack (1 = Yes, 0 = No)")
plt.yticks([0, 1], ["No", "Yes"])
plt.legend(title="Gender")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'ex6_age_vs_panic.png'))
plt.close()
print("Saved: ex6_age_vs_panic.png")

print("\nAll exercises complete. Plots saved to:", SCRIPT_DIR)
