import pandas as pd
from transformers import pipeline
import torch

df = pd.read_csv("data/processed/reviews_bilingual_only.csv")
print(f"Loaded {len(df)} bilingual reviews")

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

en_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    truncation=True,
    max_length=512,
)

fr_sentiment = pipeline(
    "sentiment-analysis",
    model="philschmid/pt-tblard-tf-allocine",
    device=device,
    truncation=True,
    max_length=512,
)

def analyze_batch(texts, pipe):
    results = pipe(texts, batch_size=32)
    return [r["label"] for r in results], [r["score"] for r in results]

en_df = df[df["final_lang"] == "en"].copy()
fr_df = df[df["final_lang"] == "fr"].copy()

print(f"\nRunning English sentiment on {len(en_df)} reviews...")
en_labels, en_scores = analyze_batch(en_df["content"].astype(str).tolist(), en_sentiment)
en_df["sentiment_label"] = en_labels
en_df["sentiment_score"] = en_scores

print(f"Running French sentiment on {len(fr_df)} reviews...")
fr_labels, fr_scores = analyze_batch(fr_df["content"].astype(str).tolist(), fr_sentiment)
fr_df["sentiment_label"] = fr_labels
fr_df["sentiment_score"] = fr_scores

result = pd.concat([en_df, fr_df], ignore_index=True)
result.to_csv("data/processed/reviews_with_sentiment.csv", index=False)

print("\nEnglish sentiment distribution:")
print(en_df["sentiment_label"].value_counts())
print("\nFrench sentiment distribution:")
print(fr_df["sentiment_label"].value_counts())