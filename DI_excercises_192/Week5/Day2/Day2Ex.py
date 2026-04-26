import numpy as np

# Exercise - Statistical Analysis
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=50)  # normal distribution: mean=100, std=15, 50 values

mean_val   = np.mean(data)
median_val = np.median(data)
std_val    = np.std(data)
q1         = np.percentile(data, 25)
q3         = np.percentile(data, 75)

print(f"Mean:    {mean_val:.2f}")
print(f"Median:  {median_val:.2f}")
print(f"Std Dev: {std_val:.2f}")
print(f"Q1 (25th): {q1:.2f}")
print(f"Q3 (75th): {q3:.2f}")

# Q4: Are mean and median close?
diff = abs(mean_val - median_val)
print(f"\nDifference between mean and median: {diff:.2f}")
if diff < 2:
    print("Mean and median are very close -> the distribution is approximately symmetric (normal).")
else:
    print("Mean and median differ -> some skew is present in this sample.")
