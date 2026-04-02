"""
src/models/fusion.py  —  Step 5: Fusion Score
----------------------------------------------
Combines text model probabilities and temporal anomaly scores
into one final Hype Score (0–100) per product.

Formula (configurable weights):
    raw_score = w_text  × mean(fake_prob per product)
              + w_time  × temporal_score
              + w_burst × five_star_pct      (from temporal features)
              + w_var   × rating_std_penalty

Final Hype Score = clip(raw_score × 100, 0, 100)

Usage:
    from src.models.fusion import compute_hype_scores
    df = compute_hype_scores()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.models.temporal_model import build_temporal_features


# ── Default weights (must sum to 1.0) ─────────────────────────
WEIGHTS = {
    "text_prob":      0.45,   # average fake_prob from TF-IDF model
    "temporal":       0.35,   # temporal anomaly score from Isolation Forest
    "five_star_pct":  0.10,   # fraction of reviews that are 5-star
    "burst_ratio":    0.10,   # fraction of burst days
}

SAVE_PATH = "data/features/hype_scores.csv"


def compute_hype_scores(
    text_probs_path:     str = "data/features/text_probs.csv",
    temporal_scores_path:str = "data/features/temporal_scores.csv",
    clean_csv_path:      str = "data/processed/clean_reviews.csv",
    save_path:           str = SAVE_PATH,
    weights:             dict = None,
    verbose:             bool = True,
) -> pd.DataFrame:
    """
    Merge text and temporal scores, compute Hype Score per product.

    Returns
    -------
    DataFrame with columns:
        product_id, hype_score, risk_level,
        text_score, temporal_score, five_star_pct, burst_ratio,
        total_reviews
    """
    w = weights or WEIGHTS
    assert abs(sum(w.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # ── 1. Load text probabilities → aggregate per product ───
    text_df = pd.read_csv(text_probs_path)
    text_agg = (
        text_df.groupby("product_id")["fake_prob"]
        .agg(text_score=("mean"), review_count=("count"))
        .reset_index()
    )

    # ── 2. Load temporal scores ───────────────────────────────
    temp_df = pd.read_csv(temporal_scores_path)

    # ── 3. Load extra temporal features (five_star_pct, burst) ─
    clean_df = pd.read_csv(clean_csv_path, parse_dates=["date"])
    feat_df  = build_temporal_features(clean_df).reset_index()
    feat_df  = feat_df[["product_id", "five_star_pct", "burst_ratio",
                         "total_reviews", "daily_max", "daily_mean",
                         "rating_std"]]

    # ── 4. Merge all signals ──────────────────────────────────
    merged = text_agg.merge(temp_df, on="product_id", how="inner")
    merged = merged.merge(feat_df, on="product_id", how="left")

    # ── 5. Normalise each component to [0, 1] ─────────────────
    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-9 else pd.Series(0.5, index=s.index)

    merged["text_score_n"]    = minmax(merged["text_score"])
    merged["temporal_n"]      = minmax(merged["temporal_score"])
    merged["five_star_pct_n"] = minmax(merged["five_star_pct"].fillna(0))
    merged["burst_ratio_n"]   = minmax(merged["burst_ratio"].fillna(0))

    # ── 6. Weighted fusion score ──────────────────────────────
    merged["raw_score"] = (
        w["text_prob"]     * merged["text_score_n"]
      + w["temporal"]      * merged["temporal_n"]
      + w["five_star_pct"] * merged["five_star_pct_n"]
      + w["burst_ratio"]   * merged["burst_ratio_n"]
    )

    # ── 7. Scale to 0–100 ─────────────────────────────────────
    merged["hype_score"] = np.clip(merged["raw_score"] * 100, 0, 100).round(1)

    # ── 8. Risk level ─────────────────────────────────────────
    def risk_label(score: float) -> str:
        if score < 33:   return "Low"
        elif score < 66: return "Medium"
        return "High"

    merged["risk_level"] = merged["hype_score"].apply(risk_label)

    # ── 9. Select and sort output ─────────────────────────────
    out = merged[[
        "product_id", "hype_score", "risk_level",
        "text_score", "temporal_score",
        "five_star_pct", "burst_ratio",
        "total_reviews", "daily_max", "daily_mean", "rating_std",
    ]].sort_values("hype_score", ascending=False).reset_index(drop=True)

    out["text_score"]   = out["text_score"].round(4)
    out["temporal_score"] = out["temporal_score"].round(4)

    # ── 10. Save ─────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)

    if verbose:
        print(f"[Fusion] Products scored  : {len(out)}")
        print(f"[Fusion] High risk        : {(out['risk_level']=='High').sum()}")
        print(f"[Fusion] Medium risk      : {(out['risk_level']=='Medium').sum()}")
        print(f"[Fusion] Low risk         : {(out['risk_level']=='Low').sum()}")
        print(f"[Fusion] Hype score range : "
              f"{out['hype_score'].min()} – {out['hype_score'].max()}")
        print(f"[Fusion] Saved to         : {save_path}")

    return out
