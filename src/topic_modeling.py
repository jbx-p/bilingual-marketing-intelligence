import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

df = pd.read_csv("data/processed/reviews_with_sentiment.csv")

MIN_LEN_FOR_TOPICS = 25

en_docs = df[(df["final_lang"] == "en") & (df["content"].str.len() >= MIN_LEN_FOR_TOPICS)]
fr_docs = df[(df["final_lang"] == "fr") & (df["content"].str.len() >= MIN_LEN_FOR_TOPICS)]

print(f"English documents for topic modeling: {len(en_docs)}")
print(f"French documents for topic modeling: {len(fr_docs)}")

en_embedder = SentenceTransformer("all-MiniLM-L6-v2")
fr_embedder = SentenceTransformer("dangvantuan/sentence-camembert-base")

en_umap = UMAP(random_state=42)
fr_umap = UMAP(random_state=42)

en_hdbscan = HDBSCAN(min_cluster_size=150, min_samples=10, cluster_selection_method="leaf", prediction_data=True)
fr_hdbscan = HDBSCAN(min_cluster_size=40, min_samples=5, cluster_selection_method="leaf", prediction_data=True)

en_topic_model = BERTopic(embedding_model=en_embedder, umap_model=en_umap, hdbscan_model=en_hdbscan, verbose=True)
fr_topic_model = BERTopic(embedding_model=fr_embedder, umap_model=fr_umap, hdbscan_model=fr_hdbscan, verbose=True)

print("\nFitting English topic model (this will take a while on CPU)...")
en_texts = en_docs["content"].astype(str).tolist()
en_topics, en_probs = en_topic_model.fit_transform(en_texts)

print("\nFitting French topic model...")
fr_texts = fr_docs["content"].astype(str).tolist()
fr_topics, fr_probs = fr_topic_model.fit_transform(fr_texts)

en_docs = en_docs.copy()
fr_docs = fr_docs.copy()
en_docs["topic"] = en_topics
fr_docs["topic"] = fr_topics

en_docs.to_csv("data/processed/en_reviews_with_topics.csv", index=False)
fr_docs.to_csv("data/processed/fr_reviews_with_topics.csv", index=False)

en_topic_model.save("outputs/en_topic_model", serialization="safetensors", save_ctfidf=True)
fr_topic_model.save("outputs/fr_topic_model", serialization="safetensors", save_ctfidf=True)

print("\nSaved topic-tagged data and models.")

french_stopwords = [
    "au","aux","avec","ce","ces","dans","de","des","du","elle","en","et","eux","il",
    "je","la","le","les","leur","lui","ma","mais","me","même","mes","moi","mon","ne",
    "nos","notre","nous","on","ou","par","pas","pour","qu","que","qui","sa","se",
    "ses","son","sur","ta","te","tes","toi","ton","tu","un","une","vos","votre",
    "vous","c","d","j","l","à","m","n","s","t","y","été","étée","étées","étés",
    "étant","suis","es","est","sommes","êtes","sont","serai","seras","sera",
    "avais","avait","avions","aviez","avaient","fais","fait","faisons","faites","font",
    "très","plus","bien","tout","toute","tous","toutes"
]

en_vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
fr_vectorizer = CountVectorizer(stop_words=french_stopwords, ngram_range=(1, 2))

en_topic_model.update_topics(en_texts, vectorizer_model=en_vectorizer)
fr_topic_model.update_topics(fr_texts, vectorizer_model=fr_vectorizer)

print("\n=== ENGLISH TOPICS (cleaned) ===")
print(en_topic_model.get_topic_info()[["Topic","Count","Name"]].head(20).to_string())
print("\n=== FRENCH TOPICS (cleaned) ===")
print(fr_topic_model.get_topic_info()[["Topic","Count","Name"]].head(20).to_string())

en_topic_model.save("outputs/en_topic_model", serialization="safetensors", save_ctfidf=True)
fr_topic_model.save("outputs/fr_topic_model", serialization="safetensors", save_ctfidf=True)

print("\nSaved cleaned topic models (overwriting previous versions).")