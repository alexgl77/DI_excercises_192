import pandas as pd
import seaborn as sns
import numpy as np

# Load the restaurant tips dataset
tips = sns.load_dataset('tips')

# Quick inspection - what are we working with?
print(f"Dataset: {tips.shape[0]} rows, {tips.shape[1]} columns")
print(f"\nColumns: {list(tips.columns)}")
tips.head(10)

# Check for missing data
print("Missing values:")
print(tips.isna().sum())

# Let's make it realistic - add some missing data
tips_messy = tips.copy()

# Randomly remove some tip values (simulating data entry errors)
np.random.seed(42)
missing_idx = np.random.choice(tips_messy.index, size=15, replace=False)
tips_messy.loc[missing_idx, 'tip'] = np.nan

print(f"Now we have {tips_messy['tip'].isna().sum()} missing tip values")
print("\nThis is your working dataset: tips_messy")

# ─── Q1: Which day has the highest AVERAGE tip? ───────────────────────────────
avg_tip_by_day = tips_messy.groupby('day')['tip'].mean().sort_values(ascending=False)
print("Q1 - Average tip by day:")
print(avg_tip_by_day)
print(f"\nBest day: {avg_tip_by_day.index[0]} with ${avg_tip_by_day.iloc[0]:.2f} avg tip\n")

# ─── Q2: Dinner parties of 4+ people who tipped over $5 ──────────────────────
mask = (tips_messy['time'] == 'Dinner') & (tips_messy['size'] >= 4) & (tips_messy['tip'] > 5)
big_dinner_tippers = tips_messy.loc[mask, ['day', 'time', 'size', 'total_bill', 'tip']]
print("Q2 - Dinner parties (4+ people, tip > $5):")
print(big_dinner_tippers)
print(f"\nTotal: {len(big_dinner_tippers)} parties\n")

# ─── Q3: Top 10 tippers and what they have in common ─────────────────────────
top10 = tips_messy.sort_values('tip', ascending=False).head(10)
print("Q3 - Top 10 tippers:")
print(top10[['day', 'time', 'size', 'total_bill', 'tip', 'smoker', 'sex']])
print("\nCommon traits:")
print(f"  Time:   {top10['time'].value_counts().idxmax()} dominates")
print(f"  Day:    {top10['day'].value_counts().idxmax()} dominates")
print(f"  Smoker: {top10['smoker'].value_counts().idxmax()}")
print(f"  Avg size: {top10['size'].mean():.1f} people\n")

# ─── Q4: Fill missing tip values ─────────────────────────────────────────────
# Fill with median tip per day (more reasonable than overall median)
tips_clean = tips_messy.copy()
tips_clean['tip'] = tips_messy.groupby('day')['tip'].transform(
    lambda x: x.fillna(x.median())
)

print("Q4 - Missing values after cleaning:")
print(tips_clean['tip'].isna().sum(), "missing values remaining")
