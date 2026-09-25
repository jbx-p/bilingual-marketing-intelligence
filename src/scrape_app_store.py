from app_store_scraper import AppStore
import pandas as pd

APPS = {
    "tim_hortons": {"app_name": "tim-hortons", "app_id": "1143883086"},
    "air_canada": {"app_name": "air-canada", "app_id": "326459697"},
    "rbc": {"app_name": "rbc-mobile", "app_id": "1169558948"},
}

all_reviews = []

for brand, info in APPS.items():
    try:
        app = AppStore(country="ca", app_name=info["app_name"], app_id=info["app_id"])
        app.review(how_many=2000)
        for r in app.reviews:
            r["brand"] = brand
            r["platform"] = "app_store"
        all_reviews.extend(app.reviews)
        print(f"{brand}: pulled {len(app.reviews)} reviews")
    except Exception as e:
        print(f"FAILED {brand}: {e}")

df = pd.DataFrame(all_reviews)
print(f"\nTotal raw reviews collected: {len(df)}")
df.to_csv("data/raw/app_store_reviews_raw.csv", index=False)
print("Saved to data/raw/app_store_reviews_raw.csv")