import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm, ttest_ind, f_oneway, binom, pearsonr, spearmanr
from scipy.stats import linregress

# ─── Exercise 1: Basic Usage of SciPy ────────────────────────────────────────
print("=== Exercise 1: SciPy Version ===")
print(f"SciPy version: {scipy.__version__}")

# ─── Exercise 2: Descriptive Statistics ──────────────────────────────────────
print("\n=== Exercise 2: Descriptive Statistics ===")

data = [12, 15, 13, 12, 18, 20, 22, 21]

mean     = stats.tmean(data)
median   = np.median(data)
variance = stats.tvar(data)
std_dev  = stats.tstd(data)

print(f"Data:               {data}")
print(f"Mean:               {mean:.2f}")
print(f"Median:             {median:.2f}")
print(f"Variance:           {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# ─── Exercise 3: Normal Distribution ─────────────────────────────────────────
print("\n=== Exercise 3: Normal Distribution ===")

mean_val, std_val = 50, 10
x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 300)
y = norm.pdf(x, mean_val, std_val)

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='steelblue', linewidth=2)
plt.fill_between(x, y, alpha=0.2, color='steelblue')
plt.title(f'Normal Distribution (mean={mean_val}, std={std_val})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('normal_distribution.png')
plt.close()
print(f"Normal distribution plotted — mean={mean_val}, std={std_val}")
print("Saved: normal_distribution.png")

# ─── Exercise 4: T-Test ───────────────────────────────────────────────────────
print("\n=== Exercise 4: T-Test ===")

np.random.seed(42)
data1 = np.random.normal(50, 10, 100)
data2 = np.random.normal(60, 10, 100)

t_stat, p_value = ttest_ind(data1, data2)

print(f"Data1 — mean: {data1.mean():.2f}, std: {data1.std():.2f}")
print(f"Data2 — mean: {data2.mean():.2f}, std: {data2.std():.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_value:.6f}")

if p_value < 0.05:
    print("Result: The two groups are SIGNIFICANTLY different (p < 0.05)")
else:
    print("Result: No significant difference between groups (p >= 0.05)")

# ─── Exercise 5: Linear Regression ───────────────────────────────────────────
print("\n=== Exercise 5: Linear Regression ===")

house_sizes  = [50, 70, 80, 100, 120]
house_prices = [150000, 200000, 210000, 250000, 280000]

slope, intercept, r_value, p_value, std_err = linregress(house_sizes, house_prices)

print(f"Slope:     {slope:.2f}  (price increase per m²)")
print(f"Intercept: {intercept:.2f}")
print(f"R²:        {r_value**2:.4f}")

# Predict price for 90 m²
size_to_predict = 90
predicted_price = slope * size_to_predict + intercept
print(f"\nPredicted price for {size_to_predict} m²: ${predicted_price:,.2f}")

print(f"\nInterpretation:")
print(f"  For every additional m², the price increases by ${slope:,.2f}.")
print(f"  The intercept (${intercept:,.2f}) is the theoretical base price at 0 m² (not meaningful alone).")

# Plot
x_line = np.linspace(40, 130, 100)
y_line = slope * x_line + intercept

plt.figure(figsize=(7, 4))
plt.scatter(house_sizes, house_prices, color='steelblue', zorder=5, label='Data')
plt.plot(x_line, y_line, color='tomato', linewidth=2, label='Regression line')
plt.scatter([size_to_predict], [predicted_price], color='green', zorder=6,
            s=100, marker='*', label=f'Prediction: ${predicted_price:,.0f}')
plt.title('House Size vs Price — Linear Regression')
plt.xlabel('Size (m²)')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression.png')
plt.close()
print("Saved: linear_regression.png")

# ─── Exercise 6: ANOVA ────────────────────────────────────────────────────────
print("\n=== Exercise 6: ANOVA ===")

fertilizer_1 = [5, 6, 7, 6, 5]
fertilizer_2 = [7, 8, 7, 9, 8]
fertilizer_3 = [4, 5, 4, 3, 4]

f_value, p_value = f_oneway(fertilizer_1, fertilizer_2, fertilizer_3)

print(f"F-value: {f_value:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("Result: The fertilizers have SIGNIFICANTLY different effects (p < 0.05)")
else:
    print("Result: No significant difference between fertilizers (p >= 0.05)")

print(f"\nQ: What if P-value > 0.05?")
print(f"   We would fail to reject the null hypothesis — meaning there is not enough")
print(f"   statistical evidence to conclude the fertilizers produce different results.")

# ─── Exercise 7: Binomial Distribution (Optional) ────────────────────────────
print("\n=== Exercise 7: Binomial Distribution (Optional) ===")

# Probability of exactly 5 heads in 10 coin flips (p=0.5)
n, p, k = 10, 0.5, 5
prob_exactly_5 = binom.pmf(k, n, p)
print(f"P(exactly {k} heads in {n} flips): {prob_exactly_5:.4f} ({prob_exactly_5*100:.2f}%)")

# Plot full distribution
x_binom = np.arange(0, n + 1)
y_binom = binom.pmf(x_binom, n, p)

plt.figure(figsize=(7, 4))
plt.bar(x_binom, y_binom, color='steelblue', edgecolor='white', alpha=0.8)
plt.bar(k, binom.pmf(k, n, p), color='tomato', edgecolor='white', label=f'k={k}')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('binomial_distribution.png')
plt.close()
print("Saved: binomial_distribution.png")

# ─── Exercise 8: Correlation Coefficients (Optional) ─────────────────────────
print("\n=== Exercise 8: Correlation Coefficients (Optional) ===")

data = pd.DataFrame({
    'age':    [23, 25, 30, 35, 40],
    'income': [35000, 40000, 50000, 60000, 70000]
})

pearson_r,  pearson_p  = pearsonr(data['age'], data['income'])
spearman_r, spearman_p = spearmanr(data['age'], data['income'])

print(f"Pearson  r: {pearson_r:.4f}  (p={pearson_p:.4f})")
print(f"Spearman r: {spearman_r:.4f}  (p={spearman_p:.4f})")
print(f"\nInterpretation:")
print(f"  Pearson  measures LINEAR correlation — r={pearson_r:.2f} means strong positive linear relationship.")
print(f"  Spearman measures RANK correlation — r={spearman_r:.2f} means strong monotonic relationship.")
print(f"  Both confirm: as age increases, income increases consistently.")
