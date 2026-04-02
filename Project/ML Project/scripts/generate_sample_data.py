"""
scripts/generate_sample_data.py
--------------------------------
Generates a realistic synthetic Amazon-style review dataset
for development and testing when you don't have the real dataset.

Run:  python scripts/generate_sample_data.py
Output: data/raw/amazon_reviews.csv  (2000 reviews, 20 products)
"""

import random
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Vocabulary ────────────────────────────────────────────────
GENUINE_PHRASES = [
    "Great product, works exactly as described.",
    "Very solid build quality, happy with my purchase.",
    "Shipped fast, packaging was good, product works well.",
    "Used it for two weeks now, still holding up.",
    "Good value for the price, would buy again.",
    "It's okay, does the job but nothing special.",
    "Broke after one month, disappointed.",
    "Not what I expected based on the photos.",
    "Customer service was helpful when I had an issue.",
    "My kids love it, perfect size for them.",
    "Average quality, expected better at this price point.",
    "Works fine but the instructions are terrible.",
    "Returned it, color was completely different.",
    "Best purchase I've made this year, highly recommend.",
    "The smell is a bit odd but goes away after a day.",
]

FAKE_PHRASES = [
    "AMAZING!!! Best product EVER!!! Buy it NOW!!!",
    "5 stars! Perfect in every way! Must buy immediately!",
    "Absolutely incredible, changed my life completely!",
    "This is the greatest thing I have ever purchased!!!",
    "WOW! Exceeded ALL expectations! 10/10 recommend!!!",
    "Phenomenal quality! Buy it today you won't regret it!",
    "Outstanding product! Perfect! Amazing! Buy buy buy!",
    "Superb! Excellent! World class! Highly recommended!",
    "Life changing product! Everyone should own one!!!",
    "Marvelous! Stunning quality! Will buy 10 more!!!",
]

PRODUCT_NAMES = [
    "Bluetooth Speaker X1", "LED Desk Lamp Pro", "Yoga Mat Ultra",
    "Stainless Steel Water Bottle", "Mechanical Keyboard RGB",
    "Wireless Mouse Slim", "Phone Stand Adjustable", "USB-C Hub 7-Port",
    "Neck Massager Electric", "Air Purifier Mini",
    "Smart Watch Fitness", "Portable Charger 20000mAh", "Foam Roller Set",
    "Coffee Grinder Manual", "Ring Light 10-inch",
    "Laptop Stand Aluminum", "Cable Organizer Box", "Silk Sleep Mask",
    "Resistance Bands Set", "Bamboo Cutting Board",
]


def make_reviewer_id(fake_burst=False):
    if fake_burst:
        # Fake accounts: short IDs clustered together
        return f"R{random.randint(9000, 9200):04d}"
    return f"R{random.randint(1000, 8999):04d}"


def generate_dataset(n_products=20, total_reviews=2000, hype_product_fraction=0.35):
    """
    Build synthetic review data.
    hype_product_fraction = fraction of products with injected fake hype.
    """
    records = []
    products = PRODUCT_NAMES[:n_products]
    hype_products = set(random.sample(products, k=int(n_products * hype_product_fraction)))

    reviews_per_product = total_reviews // n_products
    base_date = datetime(2024, 1, 1)

    review_id = 1
    for product in products:
        is_hype = product in hype_products

        # ── Normal organic reviews (spread over 6 months) ────
        organic_count = reviews_per_product if not is_hype else int(reviews_per_product * 0.5)
        for _ in range(organic_count):
            days_offset = random.randint(0, 180)
            hour_offset = random.randint(0, 23)
            date = base_date + timedelta(days=days_offset, hours=hour_offset)
            text = random.choice(GENUINE_PHRASES)
            rating = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 15, 35, 37])[0]
            records.append({
                "review_id":   f"REV{review_id:05d}",
                "product_id":  product,
                "reviewer_id": make_reviewer_id(fake_burst=False),
                "review_text": text,
                "rating":      rating,
                "date":        date.strftime("%Y-%m-%d %H:%M:%S"),
                "is_fake":     0,
            })
            review_id += 1

        # ── Fake hype burst reviews for hype products ─────────
        if is_hype:
            burst_count = int(reviews_per_product * 0.55)
            # All pile up in a short 3-day burst
            burst_start = base_date + timedelta(days=random.randint(30, 120))
            for _ in range(burst_count):
                minutes_offset = random.randint(0, 3 * 24 * 60)  # 3-day window
                date = burst_start + timedelta(minutes=minutes_offset)
                text = random.choice(FAKE_PHRASES)
                rating = 5  # fake reviews are always 5 stars
                records.append({
                    "review_id":   f"REV{review_id:05d}",
                    "product_id":  product,
                    "reviewer_id": make_reviewer_id(fake_burst=True),
                    "review_text": text,
                    "rating":      rating,
                    "date":        date.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_fake":     1,
                })
                review_id += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df, hype_products


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out_path = "data/raw/amazon_reviews.csv"

    print("Generating synthetic review dataset ...")
    df, hype_products = generate_dataset()

    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Total reviews  : {len(df)}")
    print(f"Total products : {df['product_id'].nunique()}")
    print(f"Fake reviews   : {df['is_fake'].sum()} ({df['is_fake'].mean()*100:.1f}%)")
    print(f"Hype products  : {sorted(hype_products)}")
    print(f"\nSample:")
    print(df.head(5).to_string(index=False))
