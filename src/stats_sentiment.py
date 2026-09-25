import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

df = pd.read_csv("data/processed/reviews_with_sentiment.csv")
print(f"Loaded {len(df)} reviews\n")

# --- Test 1: Overall chi-square, language vs sentiment ---
overall_table = pd.crosstab(df["final_lang"], df["sentiment_label"])
print("=== Overall contingency table ===")
print(overall_table)

chi2, p, dof, expected = chi2_contingency(overall_table)
n = overall_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(overall_table.shape) - 1)))

print(f"\nChi-square: {chi2:.3f}, p-value: {p:.6f}, dof: {dof}")
print(f"Cramér's V (effect size): {cramers_v:.4f}")

# --- Test 2: Per-brand chi-square (controls for brand mix confound) ---
print("\n\n=== Per-brand breakdown ===")
for brand in df["brand"].unique():
    sub = df[df["brand"] == brand]
    table = pd.crosstab(sub["final_lang"], sub["sentiment_label"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        print(f"\n{brand}: insufficient data for a valid test, skipping")
        continue
    chi2_b, p_b, dof_b, _ = chi2_contingency(table)
    n_b = table.sum().sum()
    v_b = np.sqrt(chi2_b / (n_b * (min(table.shape) - 1)))
    print(f"\n{brand}:")
    print(table)
    print(f"Chi-square: {chi2_b:.3f}, p-value: {p_b:.6f}, Cramér's V: {v_b:.4f}")

# --- Test 3: Two-proportion z-test with confidence interval on % negative ---
def prop_ci_diff(neg1, n1, neg2, n2, label1, label2):
    p1, p2 = neg1 / n1, neg2 / n2
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    z = diff / se
    print(f"\n{label1} negative rate: {p1:.4f} ({neg1}/{n1})")
    print(f"{label2} negative rate: {p2:.4f} ({neg2}/{n2})")
    print(f"Difference: {diff:.4f}  (95% CI: {ci_low:.4f} to {ci_high:.4f})")
    print(f"z-statistic: {z:.3f}")

print("\n\n=== Two-proportion z-test: overall negative rate, EN vs FR ===")
en = df[df["final_lang"] == "en"]
fr = df[df["final_lang"] == "fr"]
prop_ci_diff(
    (en["sentiment_label"] == "NEGATIVE").sum(), len(en),
    (fr["sentiment_label"] == "NEGATIVE").sum(), len(fr),
    "English", "French"
)