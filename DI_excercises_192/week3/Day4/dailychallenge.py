import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os

warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# NOTE: ipywidgets only works in Jupyter. In a .py script we replicate
# the interactive behaviour by running each function with representative values.

# ==============================================================================
# TASK 1: DATA SCOPING AND PREPARATION
# ==============================================================================
print("=" * 60)
print("TASK 1: DATA SCOPING AND PREPARATION")
print("=" * 60)

path = r"C:\Users\alexg\Downloads\superstore\Sample - Superstore.csv"
df = pd.read_csv(path, encoding='latin1')

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print()
print(df.dtypes)
print()
print(df.describe())

# --- Duplicates ---
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# --- Missing values ---
print("\nMissing values per column:")
print(df.isnull().sum())

if 'Postal Code' in df.columns:
    df['Postal Code'] = df['Postal Code'].fillna(0)

# --- Fix date types ---
date_columns = ['Order Date', 'Ship Date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

print("\nData types after conversion:")
print(df[date_columns].dtypes)

# --- Feature engineering ---
df['Profit Margin']    = (df['Profit'] / df['Sales']) * 100
df['Order Year']       = df['Order Date'].dt.year
df['Order Month']      = df['Order Date'].dt.month
df['Order Month-Year'] = df['Order Date'].dt.to_period('M')

print("\nNew features created:")
print(df[['Sales', 'Profit', 'Profit Margin', 'Order Year', 'Order Month']].head())

# ==============================================================================
# TASK 2: DEEP-DIVE EXPLORATORY ANALYSIS (Matplotlib)
# ==============================================================================
print("\n" + "=" * 60)
print("TASK 2: EXPLORATORY ANALYSIS")
print("=" * 60)

# --- Time-series: monthly sales per category ---
monthly_sales = df.groupby(['Order Month-Year', 'Category'])['Sales'].sum().reset_index()
monthly_sales['Date'] = monthly_sales['Order Month-Year'].dt.to_timestamp()

def plot_monthly_sales(category='All'):
    plt.figure(figsize=(12, 6))
    if category == 'All':
        total_monthly = df.groupby('Order Month-Year')['Sales'].sum()
        plt.plot(total_monthly.index.to_timestamp(), total_monthly.values,
                 marker='o', linewidth=2, markersize=4, color='steelblue')
        plt.title('Monthly Sales Trend - All Categories', fontsize=16, fontweight='bold')
    else:
        cat_data = monthly_sales[monthly_sales['Category'] == category]
        plt.plot(cat_data['Date'], cat_data['Sales'],
                 marker='o', linewidth=2, markersize=4, color='tomato')
        plt.title(f'Monthly Sales Trend - {category}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f'ts_sales_{category.replace(" ", "_")}.png'))
    plt.close()
    print(f"Saved: ts_sales_{category}.png")

# Run for All + each category
for cat in ['All'] + list(df['Category'].unique()):
    plot_monthly_sales(cat)

# --- Geographic: top N states ---
state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=True)

def plot_top_states(top_n=10):
    top_states = state_sales.tail(top_n)
    plt.figure(figsize=(12, max(6, top_n * 0.4)))
    bars = plt.barh(range(len(top_states)), top_states.values, color='steelblue')
    plt.yticks(range(len(top_states)), top_states.index)
    plt.xlabel('Total Sales ($)', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title(f'Top {top_n} States by Sales Performance', fontsize=16, fontweight='bold')
    max_val = top_states.max()
    for i, (state, value) in enumerate(top_states.items()):
        plt.text(value + max_val * 0.01, i,
                 f'${value:,.0f}', va='center', fontsize=9)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f'geo_top{top_n}_states.png'))
    plt.close()
    print(f"Saved: geo_top{top_n}_states.png")
    print(f"Total states: {len(state_sales)}")
    print(f"Top {top_n} states represent: ${top_states.sum():,.0f} in sales")

plot_top_states(10)
plot_top_states(20)

# ==============================================================================
# TASK 3: COMMUNICATING INSIGHTS (Seaborn)
# ==============================================================================
print("\n" + "=" * 60)
print("TASK 3: COMMUNICATING INSIGHTS")
print("=" * 60)

# --- Top 10 most profitable products ---
product_profit = (df.groupby('Product Name')['Profit']
                    .sum()
                    .sort_values(ascending=False)
                    .head(10))

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=product_profit.values, y=product_profit.index,
                 palette='viridis', orient='h')
plt.title('Top 10 Most Profitable Products\nExecutive Summary - Product Performance',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Total Profit ($)', fontsize=12, fontweight='bold')
plt.ylabel('Product Name', fontsize=12, fontweight='bold')
max_profit = product_profit.max()
for i, (product, profit) in enumerate(product_profit.items()):
    ax.text(profit + max_profit * 0.01, i,
            f'${profit:,.0f}', va='center', fontweight='bold', fontsize=9)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'top10_profitable_products.png'))
plt.close()
print("Saved: top10_profitable_products.png")

print(f"\nKey Insights:")
print(f"  Most profitable product: ${product_profit.iloc[0]:,.0f}")
print(f"  Top 10 total profit:     ${product_profit.sum():,.0f}")
print(f"  Average per top product: ${product_profit.mean():,.0f}")

# --- Discount vs Profit scatter ---
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='Discount', y='Profit', hue='Category', alpha=0.6, s=50)
sns.regplot(data=df, x='Discount', y='Profit', scatter=False,
            color='red', line_kws={'linewidth': 2, 'linestyle': '--'})
plt.title('Discount Strategy Analysis: Impact on Profitability by Category',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Discount Rate', fontsize=12, fontweight='bold')
plt.ylabel('Profit ($)', fontsize=12, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
plt.text(0.5, 50, 'Break-even line', fontsize=10, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'discount_vs_profit.png'))
plt.close()
print("Saved: discount_vs_profit.png")

high_discount = df[df['Discount'] > 0.2]
print(f"\nDiscount Analysis Insights:")
print(f"  Transactions with >20% discount:          {len(high_discount):,}")
print(f"  Avg profit for high discounts:            ${high_discount['Profit'].mean():.2f}")
print(f"  % high-discount sales with losses:        {(high_discount['Profit'] < 0).mean()*100:.1f}%")
print("\nCategory-specific discount impact:")
for category in df['Category'].unique():
    cat_data = df[df['Category'] == category]
    hd = cat_data[cat_data['Discount'] > 0.2]
    if len(hd) > 0:
        print(f"  {category}: avg profit at >20% discount = ${hd['Profit'].mean():.2f}")

# ==============================================================================
# TASK 4: METHODOLOGY AND TOOLING REVIEW
# ==============================================================================
print("\n" + "=" * 60)
print("TASK 4: MATPLOTLIB VS SEABORN COMPARISON")
print("=" * 60)

print("MATPLOTLIB STRENGTHS:")
print("  Fine-grained control over layout, annotations, and text positioning")
print("  Direct integration with ipywidgets for interactive dashboards")
print("  Precise subplot management and custom figure sizing")

print("\nSEABORN STRENGTHS:")
print("  Built-in statistical visualizations (regplot, heatmap, pairplot)")
print("  Automatic color palettes and clean default styling")
print("  Publication-ready aesthetics with minimal configuration")
print("  Easy categorical data handling (hue, col, row parameters)")

# Speed comparison
start = time.time()
plt.figure(figsize=(8, 6))
plt.plot(df.groupby('Order Year')['Sales'].sum())
plt.close()
matplotlib_time = time.time() - start

start = time.time()
plt.figure(figsize=(8, 6))
sns.lineplot(data=df.groupby('Order Year')['Sales'].sum().reset_index(),
             x='Order Year', y='Sales')
plt.close()
seaborn_time = time.time() - start

print(f"\nSPEED COMPARISON:")
print(f"  Matplotlib basic plot: {matplotlib_time:.4f} seconds")
print(f"  Seaborn equivalent:    {seaborn_time:.4f} seconds")

print("""
RECOMMENDATION:
  For rapid exploration  -> Matplotlib: faster rendering, full widget control.
  For stakeholder reports -> Seaborn: publication-ready aesthetics, built-in stats.
""")

# ==============================================================================
# TASK 5: EXECUTIVE SUMMARY
# ==============================================================================
print("=" * 60)
print("TASK 5: EXECUTIVE SUMMARY")
print("=" * 60)

total_sales  = df['Sales'].sum()
total_profit = df['Profit'].sum()
profit_margin = (total_profit / total_sales) * 100
top_state = state_sales.index[-1]
top_state_sales = state_sales.iloc[-1]
top_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False).index[0]
high_discount_loss_rate = (df[df['Discount'] > 0.2]['Profit'] < 0).mean() * 100

print(f"""
BUSINESS PERFORMANCE:
  Total Revenue:        ${total_sales:,.0f}
  Total Profit:         ${total_profit:,.0f}
  Overall Profit Margin:{profit_margin:.1f}%

GEOGRAPHIC PERFORMANCE:
  Top performing state: {top_state} (${top_state_sales:,.0f})
  Top 5 states share:   {(state_sales.tail(5).sum()/total_sales)*100:.1f}% of total sales

PRODUCT PERFORMANCE:
  Leading category:     {top_category}
  Most profitable product: {product_profit.index[0]}

DISCOUNT STRATEGY:
  {high_discount_loss_rate:.1f}% of orders with >20% discount result in losses.
  Recommended cap: 20% discount to maintain profitability.

KEY RECOMMENDATIONS:
  1. Limit Furniture discounts to 20% — high-discount furniture drives losses.
  2. Focus marketing spend on {top_state} and the top 5 states (highest ROI).
  3. Prioritize {top_category} category — highest revenue generator.
  4. Review discount approval process for orders above 20% threshold.
  5. Investigate low-margin sub-categories and consider discontinuing.
""")
