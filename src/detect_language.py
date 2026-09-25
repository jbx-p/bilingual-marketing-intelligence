import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

df = pd.read_csv("data/raw/google_play_reviews_raw.csv")
df = df[df["content"].notna()]
df = df[df["content"].str.strip().str.len() >= 3]
print(f"Rows after dropping empty/too-short content: {len(df)}")

MIN_LEN_FOR_DETECTION = 20

# Small curated word lists for short-text fallback - common short review phrases
EN_SHORT_SIGNALS = {"good", "great", "awesome", "excellent", "nice", "best", "love", "amazing",
                     "bad", "terrible", "works", "working", "app", "easy", "fast", "slow", "useless"}
FR_SHORT_SIGNALS = {"bien", "bon", "super", "genial", "génial", "excellent", "nul", "facile",
                     "rapide", "lent", "mauvais", "meilleur", "top", "parfait", "horrible"}

def classify(row):
    text = row["content"].strip()
    if len(text) >= MIN_LEN_FOR_DETECTION:
        try:
            lang = detect(text)
            return lang if lang in ("en", "fr") else "other_long"
        except LangDetectException:
            return "other_long"
    else:
        # Short text: check curated word lists first
        lowered = text.lower()
        words = set(lowered.replace("!", "").replace(".", "").split())
        if words & EN_SHORT_SIGNALS:
            return "en_short_fallback"
        if words & FR_SHORT_SIGNALS:
            return "fr_short_fallback"
        # No match: fall back to which locale it was queried under
        return f"{row['query_lang']}_short_querylang_fallback"

df["detected_lang"] = df.apply(classify, axis=1)

print("\nFull classification breakdown:")
print(df["detected_lang"].value_counts())

# Consolidate into final en/fr labels, tracking the method used
def consolidate(lang):
    if lang in ("en", "en_short_fallback", "en_short_querylang_fallback"):
        return "en"
    if lang in ("fr", "fr_short_fallback", "fr_short_querylang_fallback"):
        return "fr"
    return "other"

df["final_lang"] = df["detected_lang"].apply(consolidate)
df["lang_method"] = df["detected_lang"].apply(
    lambda x: "detected" if x in ("en", "fr")
    else "short_wordlist" if "short_fallback" in x and "querylang" not in x
    else "short_querylang_fallback" if "querylang" in x
    else "excluded"
)

print("\nFinal language breakdown:")
print(df["final_lang"].value_counts())
print("\nMethod breakdown (transparency on how each label was reached):")
print(df.groupby(["final_lang", "lang_method"]).size())

df_bilingual = df[df["final_lang"].isin(["en", "fr"])].copy()
print(f"\nFinal bilingual dataset size: {len(df_bilingual)}")
print(df_bilingual.groupby(["brand", "final_lang"]).size())

df.to_csv("data/processed/reviews_with_language.csv", index=False)
df_bilingual.to_csv("data/processed/reviews_bilingual_only.csv", index=False)
print("\nSaved both files to data/processed/")