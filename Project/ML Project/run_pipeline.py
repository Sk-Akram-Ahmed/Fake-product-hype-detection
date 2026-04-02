"""
run_pipeline.py  —  End-to-end MVP Pipeline Runner
---------------------------------------------------
Runs all 4 pipeline steps in sequence:
    Step 2: Load & clean data
    Step 3: Train + score text model (TF-IDF + LR)
    Step 4: Train + score temporal model (Isolation Forest)
    Step 5: Fuse scores → hype_scores.csv

Usage:
    python run_pipeline.py                               # use sample data
    python run_pipeline.py --data data/raw/my_file.csv  # use your own CSV
    python run_pipeline.py --skip-train                  # skip retraining
"""

import sys
import time
import argparse
from pathlib import Path

# ── Make src importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader       import load_and_clean
from src.models.text_model     import TextModel
from src.models.temporal_model import TemporalModel
from src.models.fusion         import compute_hype_scores


def banner(text: str):
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}")


def step(n: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*60}")


def run_pipeline(data_path: str, skip_train: bool = False):
    total_start = time.time()

    banner("FAKE PRODUCT HYPE DETECTION — MVP PIPELINE")
    print(f"  Data source : {data_path}")
    print(f"  Skip train  : {skip_train}")

    # ── STEP 2: Load & Clean ──────────────────────────────────
    step(2, "Dataset Loading & Cleaning")
    t0      = time.time()
    clean_df = load_and_clean(
        csv_path  = data_path,
        save_path = "data/processed/clean_reviews.csv",
        verbose   = True,
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(clean_df)} clean reviews")

    # ── STEP 3: Text Model ────────────────────────────────────
    step(3, "Text Model — TF-IDF + Logistic Regression")
    t0    = time.time()
    tm    = TextModel()
    if not skip_train:
        metrics = tm.train(
            csv_path  = "data/processed/clean_reviews.csv",
            verbose   = True,
        )
        print(f"  Train metrics: {metrics}")
    text_probs = tm.predict(
        csv_path  = "data/processed/clean_reviews.csv",
        save_path = "data/features/text_probs.csv",
        verbose   = True,
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(text_probs)} review probabilities")

    # ── STEP 4: Temporal Model ────────────────────────────────
    step(4, "Temporal Model — Isolation Forest")
    t0   = time.time()
    temp = TemporalModel()
    if not skip_train:
        metrics = temp.train(
            csv_path = "data/processed/clean_reviews.csv",
            verbose  = True,
        )
        print(f"  Train metrics: {metrics}")
    temp_scores = temp.predict(
        csv_path  = "data/processed/clean_reviews.csv",
        save_path = "data/features/temporal_scores.csv",
        verbose   = True,
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(temp_scores)} product anomaly scores")

    # ── STEP 5: Fusion Score ──────────────────────────────────
    step(5, "Fusion — Combining Signals into Hype Score")
    t0    = time.time()
    hype  = compute_hype_scores(
        text_probs_path      = "data/features/text_probs.csv",
        temporal_scores_path = "data/features/temporal_scores.csv",
        clean_csv_path       = "data/processed/clean_reviews.csv",
        save_path            = "data/features/hype_scores.csv",
        verbose              = True,
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(hype)} products scored")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - total_start
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(hype[["product_id", "hype_score", "risk_level"]].to_string(index=False))

    print(f"""
  Output files:
    data/processed/clean_reviews.csv
    data/features/text_probs.csv
    data/features/temporal_scores.csv
    data/features/hype_scores.csv      <-- main output

  Launch dashboard:
    streamlit run app.py
""")
    return hype


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Fake Product Hype Detection pipeline"
    )
    parser.add_argument(
        "--data",
        default="data/raw/amazon_reviews.csv",
        help="Path to raw review CSV (default: data/raw/amazon_reviews.csv)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training, use existing saved models",
    )
    args = parser.parse_args()

    # ── Auto-generate sample data if missing ─────────────────
    if not Path(args.data).exists():
        print(f"\n[INFO] Dataset not found at: {args.data}")
        print("[INFO] Generating synthetic sample data ...")
        from scripts.generate_sample_data import generate_dataset
        import pandas as pd
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df, _ = generate_dataset()
        df.to_csv(args.data, index=False)
        print(f"[INFO] Sample data saved: {args.data}  ({len(df)} reviews)")

    run_pipeline(data_path=args.data, skip_train=args.skip_train)
