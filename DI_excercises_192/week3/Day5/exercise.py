import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style='whitegrid')

# ==============================================================================
# LOAD & PREPROCESS
# ==============================================================================
path = r"C:\Users\alexg\Downloads\superstore\Sample - Superstore.csv"
df = pd.read_csv(path, encoding='latin1')

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date']  = pd.to_datetime(df['Ship Date'])
df['Order Year'] = df['Order Date'].dt.year
df['Order Month']= df['Order Date'].dt.to_period('M')

print(f"Dataset loaded: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ==============================================================================
# Q1: Which states have the most sales?
# ==============================================================================
print("=" * 60)
print("Q1: States with the Most Sales")
print("=" * 60)

state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False)
print(state_sales.head(10).to_string())

plt.figure(figsize=(14, 6))
top20_states = state_sales.head(20)
colors = ['gold' if s in ['California', 'New York'] else 'steelblue' for s in top20_states.index]
bars = plt.bar(top20_states.index, top20_states.values, color=colors, edgecolor='white')
plt.title('Top 20 States by Total Sales', fontsize=15, fontweight='bold')
plt.xlabel('State')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q1_states_by_sales.png'))
plt.close()
print("Saved: q1_states_by_sales.png\n")

# ==============================================================================
# Q2: New York vs California - Sales and Profit
# ==============================================================================
print("=" * 60)
print("Q2: New York vs California")
print("=" * 60)

ny_ca = df[df['State'].isin(['New York', 'California'])]
comparison = ny_ca.groupby('State')[['Sales', 'Profit']].sum()
comparison['Profit Margin (%)'] = (comparison['Profit'] / comparison['Sales'] * 100).round(2)
print(comparison.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, color in zip(axes, ['Sales', 'Profit'], ['steelblue', 'seagreen']):
    bars = ax.bar(comparison.index, comparison[col], color=color, edgecolor='white', width=0.5)
    ax.set_title(f'{col} - NY vs CA', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{col} ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'${bar.get_height():,.0f}', ha='center', fontsize=10, fontweight='bold')
plt.suptitle('New York vs California: Sales & Profit', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q2_ny_vs_ca.png'))
plt.close()
print("Saved: q2_ny_vs_ca.png\n")

# ==============================================================================
# Q3: Outstanding customer in New York
# ==============================================================================
print("=" * 60)
print("Q3: Outstanding Customer in New York")
print("=" * 60)

ny_customers = (df[df['State'] == 'New York']
                .groupby('Customer Name')[['Sales', 'Profit']]
                .sum()
                .sort_values('Sales', ascending=False))
print("Top 10 NY customers by Sales:")
print(ny_customers.head(10).to_string())
top_ny = ny_customers.index[0]
print(f"\nOutstanding customer: {top_ny}")
print(f"  Sales:  ${ny_customers.loc[top_ny, 'Sales']:,.2f}")
print(f"  Profit: ${ny_customers.loc[top_ny, 'Profit']:,.2f}")

plt.figure(figsize=(12, 5))
top10_ny = ny_customers.head(10)
sns.barplot(x=top10_ny['Sales'].values, y=top10_ny.index, palette='Blues_r')
plt.title(f'Top 10 Customers in New York by Sales\nOutstanding: {top_ny}', fontsize=13, fontweight='bold')
plt.xlabel('Total Sales ($)')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q3_ny_top_customers.png'))
plt.close()
print("Saved: q3_ny_top_customers.png\n")

# ==============================================================================
# Q4: Differences among states in profitability
# ==============================================================================
print("=" * 60)
print("Q4: State Profitability Differences")
print("=" * 60)

state_profit = df.groupby('State')['Profit'].sum().sort_values()
n_losing = (state_profit < 0).sum()
n_winning = (state_profit >= 0).sum()
print(f"States with positive profit: {n_winning}")
print(f"States with negative profit: {n_losing}")
print(f"\nBottom 5 (losses):\n{state_profit.head(5).to_string()}")
print(f"\nTop 5 (profit):\n{state_profit.tail(5).to_string()}")

plt.figure(figsize=(14, 7))
colors = ['tomato' if p < 0 else 'seagreen' for p in state_profit.values]
state_profit.plot(kind='bar', color=colors, edgecolor='white')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Profit by State (Green = Profitable, Red = Loss)', fontsize=14, fontweight='bold')
plt.xlabel('State')
plt.ylabel('Total Profit ($)')
plt.xticks(rotation=90, fontsize=7)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q4_state_profitability.png'))
plt.close()
print("Saved: q4_state_profitability.png\n")

# ==============================================================================
# Q5: Pareto Principle - Customers and Profit
# ==============================================================================
print("=" * 60)
print("Q5: Pareto Principle - Customers & Profit")
print("=" * 60)

customer_profit = (df.groupby('Customer Name')['Profit']
                     .sum()
                     .sort_values(ascending=False))

# Only positive-profit customers for Pareto
positive_customers = customer_profit[customer_profit > 0]
cumulative_pct = positive_customers.cumsum() / positive_customers.sum() * 100
pct_customers   = np.arange(1, len(positive_customers) + 1) / len(positive_customers) * 100

# Find how many customers reach 80% of profit
idx_80 = (cumulative_pct >= 80).idxmax()
n_80   = list(positive_customers.index).index(idx_80) + 1
pct_80 = n_80 / len(positive_customers) * 100
print(f"Total profitable customers: {len(positive_customers)}")
print(f"Customers needed to reach 80% of profit: {n_80} ({pct_80:.1f}%)")
print(f"Pareto principle {'HOLDS' if pct_80 <= 25 else 'is approximate'} (target: ~20%)")

plt.figure(figsize=(12, 5))
plt.plot(pct_customers, cumulative_pct.values, color='steelblue', linewidth=2)
plt.axhline(80, color='tomato', linestyle='--', linewidth=1.5, label='80% profit')
plt.axvline(pct_80, color='orange', linestyle='--', linewidth=1.5, label=f'{pct_80:.0f}% customers')
plt.fill_between(pct_customers[:n_80], cumulative_pct.values[:n_80], alpha=0.15, color='steelblue')
plt.xlabel('% of Customers (sorted by profit)')
plt.ylabel('Cumulative % of Profit')
plt.title('Pareto Curve - Customers vs Profit\n(80/20 Rule Analysis)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q5_pareto_customers_profit.png'))
plt.close()
print("Saved: q5_pareto_customers_profit.png\n")

# ==============================================================================
# Q6: Top 20 Cities by Sales and Profit
# ==============================================================================
print("=" * 60)
print("Q6: Top 20 Cities by Sales and Profit")
print("=" * 60)

city_stats = df.groupby('City')[['Sales', 'Profit']].sum()
city_stats['Profit Margin (%)'] = (city_stats['Profit'] / city_stats['Sales'] * 100).round(2)

top20_city_sales  = city_stats['Sales'].sort_values(ascending=False).head(20)
top20_city_profit = city_stats['Profit'].sort_values(ascending=False).head(20)

print("Top 20 Cities by Sales:")
print(top20_city_sales.to_string())
print("\nTop 20 Cities by Profit:")
print(top20_city_profit.to_string())

# Check if any top-sales city has negative profit
top20_names = top20_city_sales.index.tolist()
city_margins = city_stats.loc[top20_names, 'Profit Margin (%)'].sort_values()
print(f"\nProfit margins among top-20-sales cities:")
print(city_margins.to_string())

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, data, title, color in zip(
        axes,
        [top20_city_sales, top20_city_profit],
        ['Top 20 Cities by Sales', 'Top 20 Cities by Profit'],
        ['steelblue', 'seagreen']):
    bar_colors = ['tomato' if v < 0 else color for v in data.values]
    ax.barh(data.index[::-1], data.values[::-1], color=bar_colors[::-1], edgecolor='white')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Amount ($)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    ax.axvline(0, color='black', linewidth=0.8)
plt.suptitle('Top 20 Cities: Sales vs Profit Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q6_top20_cities.png'))
plt.close()
print("Saved: q6_top20_cities.png\n")

# ==============================================================================
# Q7: Top 20 Customers by Sales
# ==============================================================================
print("=" * 60)
print("Q7: Top 20 Customers by Sales")
print("=" * 60)

top20_customers = (df.groupby('Customer Name')['Sales']
                     .sum()
                     .sort_values(ascending=False)
                     .head(20))
print(top20_customers.to_string())

plt.figure(figsize=(12, 7))
sns.barplot(x=top20_customers.values, y=top20_customers.index, palette='Blues_r')
plt.title('Top 20 Customers by Total Sales', fontsize=14, fontweight='bold')
plt.xlabel('Total Sales ($)')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q7_top20_customers_sales.png'))
plt.close()
print("Saved: q7_top20_customers_sales.png\n")

# ==============================================================================
# Q8: Cumulative Sales Curve - Pareto on Customers & Sales
# ==============================================================================
print("=" * 60)
print("Q8: Pareto Principle - Customers & Sales")
print("=" * 60)

all_customer_sales = (df.groupby('Customer Name')['Sales']
                        .sum()
                        .sort_values(ascending=False))
cum_sales_pct = all_customer_sales.cumsum() / all_customer_sales.sum() * 100
pct_cust      = np.arange(1, len(all_customer_sales) + 1) / len(all_customer_sales) * 100

idx_80s  = (cum_sales_pct >= 80).idxmax()
n_80s    = list(all_customer_sales.index).index(idx_80s) + 1
pct_80s  = n_80s / len(all_customer_sales) * 100
print(f"Total customers: {len(all_customer_sales)}")
print(f"Customers to reach 80% of sales: {n_80s} ({pct_80s:.1f}%)")
print(f"Pareto principle {'HOLDS' if pct_80s <= 25 else 'does NOT strictly hold'} for sales")

plt.figure(figsize=(12, 5))
plt.plot(pct_cust, cum_sales_pct.values, color='steelblue', linewidth=2)
plt.axhline(80, color='tomato',  linestyle='--', linewidth=1.5, label='80% of sales')
plt.axvline(pct_80s, color='orange', linestyle='--', linewidth=1.5, label=f'{pct_80s:.0f}% of customers')
plt.fill_between(pct_cust[:n_80s], cum_sales_pct.values[:n_80s], alpha=0.15, color='steelblue')
plt.xlabel('% of Customers (sorted by sales desc.)')
plt.ylabel('Cumulative % of Sales')
plt.title('Cumulative Sales Curve by Customer\n(Pareto / 80-20 Analysis)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q8_pareto_customers_sales.png'))
plt.close()
print("Saved: q8_pareto_customers_sales.png\n")

# ==============================================================================
# MARKETING STRATEGY RECOMMENDATIONS
# ==============================================================================
print("=" * 60)
print("MARKETING STRATEGY RECOMMENDATIONS")
print("=" * 60)

top3_states = state_sales.head(3).index.tolist()
top3_cities = top20_city_sales.head(3).index.tolist()
losing_states = state_profit[state_profit < 0].sort_values().head(5).index.tolist()

print(f"""
PRIORITY STATES FOR MARKETING:
  Focus on: {', '.join(top3_states)}
  These generate the highest revenue and should receive the most ad spend.

PRIORITY CITIES:
  Top cities: {', '.join(top3_cities)}
  Invest in local campaigns, promotions, and logistics optimization here.

STATES TO REVIEW (LOSSES):
  {', '.join(losing_states)}
  These states are unprofitable. Review pricing, discount policies,
  and shipping costs before expanding marketing investment there.

CUSTOMER STRATEGY (PARETO):
  {pct_80s:.0f}% of customers drive 80% of sales.
  {pct_80:.0f}% of customers drive 80% of profit.
  -> Build a VIP loyalty program targeting the top {pct_80s:.0f}% of customers.
  -> Upsell and cross-sell campaigns to existing high-value customers are
     more cost-effective than acquiring new ones.

NEW YORK vs CALIFORNIA:
  Both are top revenue states but margins differ.
  -> California: higher revenue, prioritize volume-based campaigns.
  -> New York: focus on high-value customers like {top_ny}.
""")
