"""
src/data/loader.py  —  Step 2: Dataset Loader
----------------------------------------------
Loads a raw Amazon/Yelp-style CSV, validates its schema,
cleans text and date columns, and saves cleaned output.

Expected CSV columns (flexible — see COLUMN_MAP below):
    review_id, product_id, reviewer_id, review_text, rating, date, [is_fake]

Usage:
    from src.data.loader import load_and_clean
    df = load_and_clean("data/raw/amazon_reviews.csv")
"""

import re
import pandas as pd
from pathlib import Path


# ── Flexible column aliases ───────────────────────────────────
# Maps whatever names the CSV uses → our standard names
COLUMN_MAP = {
    # review text
    "reviewtext": "review_text",
    "review":     "review_text",
    "text":       "review_text",
    "body":       "review_text",
    "content":    "review_text",
    # product id
    "asin":           "product_id",
    "productid":      "product_id",
    "product":        "product_id",
    "item_id":        "product_id",
    # reviewer id
    "reviewerid":     "reviewer_id",
    "user_id":        "reviewer_id",
    "userid":         "reviewer_id",
    "author":         "reviewer_id",
    # rating
    "overall":    "rating",
    "stars":      "rating",
    "score":      "rating",
    # date
    "reviewtime": "date",
    "timestamp":  "date",
    "unixreviewtime": "date",
    "review_date": "date",
    "time":        "date",
    # fake label (optional)
    "label":      "is_fake",
    "fake":       "is_fake",
    "fraud":      "is_fake",
}

REQUIRED_COLUMNS = ["product_id", "review_text", "rating", "date"]


# ── Text cleaning ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightweight text cleaner:
      - lowercase
      - strip HTML tags
      - remove URLs
      - collapse whitespace
      - keep letters, digits, basic punctuation
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Main loader ───────────────────────────────────────────────

def load_and_clean(
    csv_path: str,
    save_path: str = "data/processed/clean_reviews.csv",
    min_text_length: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a raw review CSV, standardise columns, clean text and dates,
    drop bad rows, and save the result.

    Parameters
    ----------
    csv_path        : path to the raw CSV file
    save_path       : where to write the cleaned CSV
    min_text_length : drop reviews shorter than this (after cleaning)
    verbose         : print progress messages

    Returns
    -------
    pd.DataFrame with columns:
        review_id, product_id, reviewer_id, review_text,
        rating, date, is_fake (if present), text_clean
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Run first:  python scripts/generate_sample_data.py"
        )

    # ── 1. Load ────────────────────────────────────────────────
    if verbose:
        print(f"[Loader] Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if verbose:
        print(f"[Loader] Raw shape: {df.shape}")

    # ── 2. Normalise column names ──────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for old, new in COLUMN_MAP.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # ── 3. Validate required columns ──────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ── 4. Add optional columns if absent ─────────────────────
    if "review_id" not in df.columns:
        df.insert(0, "review_id", [f"REV{i:05d}" for i in range(len(df))])
    if "reviewer_id" not in df.columns:
        df["reviewer_id"] = "UNKNOWN"
    if "is_fake" not in df.columns:
        df["is_fake"] = -1          # -1 = unknown (unlabelled)

    # ── 5. Drop rows missing critical fields ──────────────────
    before = len(df)
    df.dropna(subset=["review_text", "rating", "date"], inplace=True)
    if verbose and len(df) < before:
        print(f"[Loader] Dropped {before - len(df)} rows with nulls in key fields")

    # ── 6. Clean text ─────────────────────────────────────────
    df["text_clean"] = df["review_text"].apply(clean_text)
    short_mask = df["text_clean"].str.len() < min_text_length
    if verbose and short_mask.sum():
        print(f"[Loader] Dropped {short_mask.sum()} reviews with text < {min_text_length} chars")
    df = df[~short_mask].copy()

    # ── 7. Normalise rating to float ──────────────────────────
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating"], inplace=True)
    df["rating"] = df["rating"].clip(1, 5)

    # ── 8. Parse dates ─────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    nat_count = df["date"].isna().sum()
    if nat_count:
        if verbose:
            print(f"[Loader] Dropping {nat_count} rows with unparseable dates")
        df.dropna(subset=["date"], inplace=True)

    # ── 9. Sort and reset ─────────────────────────────────────
    df.sort_values(["product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 10. Final column selection ────────────────────────────
    keep_cols = ["review_id", "product_id", "reviewer_id",
                 "review_text", "text_clean", "rating", "date", "is_fake"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # ── 11. Save ──────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    if verbose:
        print(f"[Loader] Clean shape : {df.shape}")
        print(f"[Loader] Products    : {df['product_id'].nunique()}")
        print(f"[Loader] Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
        if "is_fake" in df.columns and (df["is_fake"] >= 0).any():
            labeled = df[df["is_fake"] >= 0]
            fake_pct = labeled["is_fake"].mean() * 100
            print(f"[Loader] Fake reviews: {labeled['is_fake'].sum()} ({fake_pct:.1f}%)")
        print(f"[Loader] Saved to    : {save_path}")

    return df


# ── Quick validation report ───────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict for display in the Streamlit app."""
    labeled = df[df["is_fake"] >= 0] if "is_fake" in df.columns else df
    return {
        "total_reviews":   len(df),
        "total_products":  df["product_id"].nunique(),
        "total_reviewers": df["reviewer_id"].nunique() if "reviewer_id" in df.columns else "N/A",
        "date_range":      f"{df['date'].min().date()} to {df['date'].max().date()}",
        "avg_rating":      round(df["rating"].mean(), 2),
        "fake_pct":        round(labeled["is_fake"].mean() * 100, 1) if len(labeled) else "N/A",
        "missing_text":    int(df["text_clean"].str.len().eq(0).sum()),
    }
