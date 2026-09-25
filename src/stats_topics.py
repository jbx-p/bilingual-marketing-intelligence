import pandas as pd
from scipy.stats import chi2_contingency

en_df = pd.read_csv("data/processed/en_reviews_with_topics.csv")
fr_df = pd.read_csv("data/processed/fr_reviews_with_topics.csv")

EN_CATEGORY_MAP = {
    -1: "Other/Unclustered",
    0: "Customer Service", 46: "Meta/Rating",
    1: "Payments/Cards", 6: "Payments/Cards", 21: "Payments/Cards", 44: "Payments/Cards", 50: "Payments/Cards", 91: "Payments/Cards",
    2: "Ease of Use", 38: "Ease of Use", 43: "Ease of Use", 84: "Ease of Use", 92: "Ease of Use",
    3: "Promotions/Games", 23: "Promotions/Games", 25: "Promotions/Games", 33: "Promotions/Games", 78: "Promotions/Games", 83: "Promotions/Games",
    4: "App Performance Issues", 10: "App Performance Issues", 15: "App Performance Issues", 29: "App Performance Issues",
    30: "App Performance Issues", 37: "App Performance Issues", 40: "App Performance Issues", 54: "App Performance Issues",
    57: "App Performance Issues", 58: "App Performance Issues", 70: "App Performance Issues", 74: "App Performance Issues",
    87: "App Performance Issues", 88: "App Performance Issues", 93: "App Performance Issues", 94: "App Performance Issues",
    5: "Banking (RBC)", 9: "Banking (RBC)", 13: "Banking (RBC)", 17: "Banking (RBC)", 18: "Banking (RBC)",
    26: "Banking (RBC)", 31: "Banking (RBC)", 42: "Banking (RBC)",
    7: "Rewards/Points/Offers", 16: "Rewards/Points/Offers", 20: "Rewards/Points/Offers", 22: "Rewards/Points/Offers",
    28: "Rewards/Points/Offers", 56: "Rewards/Points/Offers", 61: "Rewards/Points/Offers", 86: "Rewards/Points/Offers",
    8: "Food/Coffee/Menu", 14: "Food/Coffee/Menu", 36: "Food/Coffee/Menu", 45: "Food/Coffee/Menu", 49: "Food/Coffee/Menu",
    53: "Food/Coffee/Menu", 55: "Food/Coffee/Menu", 59: "Food/Coffee/Menu", 66: "Food/Coffee/Menu", 67: "Food/Coffee/Menu",
    72: "Food/Coffee/Menu", 73: "Food/Coffee/Menu", 81: "Food/Coffee/Menu", 85: "Food/Coffee/Menu",
    11: "Tim Hortons Brand Love", 48: "Tim Hortons Brand Love", 65: "Tim Hortons Brand Love", 69: "Tim Hortons Brand Love",
    75: "Tim Hortons Brand Love", 76: "Tim Hortons Brand Love",
    12: "Device Compatibility", 52: "Device Compatibility", 71: "Device Compatibility",
    19: "Flights/Air Canada", 32: "Flights/Air Canada", 63: "Flights/Air Canada", 68: "Flights/Air Canada", 90: "Flights/Air Canada",
    24: "Mobile Deposit", 51: "Mobile Deposit",
    27: "Login/Account/Security", 34: "Login/Account/Security", 39: "Login/Account/Security", 60: "Login/Account/Security",
    77: "Login/Account/Security", 79: "Login/Account/Security", 80: "Login/Account/Security", 82: "Login/Account/Security", 89: "Login/Account/Security",
    35: "Permissions/Privacy", 47: "Permissions/Privacy", 62: "Permissions/Privacy",
    41: "Negative General", 64: "Negative General",
}

FR_CATEGORY_MAP = {
    -1: "Other/Unclustered",
    0: "Customer Service", 31: "Customer Service",
    1: "Rewards/Points/Offers", 9: "Rewards/Points/Offers", 27: "Rewards/Points/Offers",
    2: "App Performance Issues", 11: "App Performance Issues", 13: "App Performance Issues", 14: "App Performance Issues",
    16: "App Performance Issues", 18: "App Performance Issues", 19: "App Performance Issues", 20: "App Performance Issues",
    28: "App Performance Issues", 30: "App Performance Issues", 32: "App Performance Issues",
    3: "Food/Coffee/Menu", 15: "Food/Coffee/Menu", 29: "Food/Coffee/Menu",
    4: "Ease of Use", 21: "Ease of Use", 22: "Ease of Use",
    5: "Banking (RBC)", 6: "Banking (RBC)", 8: "Banking (RBC)",
    7: "Flights/Air Canada", 26: "Flights/Air Canada",
    10: "Tim Hortons Brand Love",
    12: "Device Compatibility",
    17: "Mobile Deposit",
    24: "Login/Account/Security",
    23: "Promotions/Games",
    25: "Other/Unclustered",
    33: "Language/Localization Issues",
}

en_df["category"] = en_df["topic"].map(EN_CATEGORY_MAP)
fr_df["category"] = fr_df["topic"].map(FR_CATEGORY_MAP)

# Safety check: confirm nothing fell through un-mapped
en_unmapped = en_df[en_df["category"].isna()]["topic"].unique()
fr_unmapped = fr_df[fr_df["category"].isna()]["topic"].unique()
print("Unmapped English topics:", en_unmapped)
print("Unmapped French topics:", fr_unmapped)

if len(en_unmapped) > 0 or len(fr_unmapped) > 0:
    print("\nWARNING: some topics are unmapped - results below are unreliable until fixed.")

print("\nEnglish category distribution:")
print(en_df["category"].value_counts())
print("\nFrench category distribution:")
print(fr_df["category"].value_counts())

en_df["lang"] = "en"
fr_df["lang"] = "fr"
combined = pd.concat([en_df[["lang", "category"]], fr_df[["lang", "category"]]])
combined = combined.dropna(subset=["category"])

table = pd.crosstab(combined["lang"], combined["category"])
print("\nFull contingency table:")
print(table.T)

chi2, p, dof, expected = chi2_contingency(table)
print(f"\nChi-square: {chi2:.3f}, p-value: {p:.6f}, dof: {dof}")

# Show each category's share within each language, for easy comparison
print("\nCategory share within each language (%):")
pct_table = pd.crosstab(combined["lang"], combined["category"], normalize="index") * 100
print(pct_table.T.round(2))

combined_full = pd.concat([
    en_df[["lang", "topic", "category"]],
    fr_df[["lang", "topic", "category"]]
])
combined_full.to_csv("data/processed/topics_categorized.csv", index=False)
print("\nSaved data/processed/topics_categorized.csv")