"""
src/models/temporal_model.py  —  Step 4: Temporal Anomaly Detector
--------------------------------------------------------------------
Groups reviews by product × day, builds a time-series feature vector
per product, and trains an Isolation Forest to detect review burst
anomalies (e.g. 40 reviews in a single day for a low-selling item).

Features per product:
    daily_mean, daily_std, daily_max, daily_cv,
    burst_count, inter_day_gap_mean, inter_day_gap_std,
    days_with_reviews, review_velocity

Usage:
    from src.models.temporal_model import TemporalModel
    model = TemporalModel()
    model.train("data/processed/clean_reviews.csv")
    scores = model.predict("data/processed/clean_reviews.csv")
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


SAVE_PATH = "models/saved/temporal_model.pkl"
SEED = 42


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one feature row per product — fully vectorised (no Python loop).

    Parameters
    ----------
    df : DataFrame with columns [product_id, date, rating]

    Returns
    -------
    DataFrame indexed by product_id with temporal features
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["day"] = df["date"].dt.normalize()   # keep as Timestamp (vectorisable)

    # ── Daily counts per (product, day) ──────────────────────
    daily = (
        df.groupby(["product_id", "day"])
        .agg(cnt=("rating", "count"), rating_mean=("rating", "mean"))
        .reset_index()
    )

    # ── Per-product daily stats ───────────────────────────────
    grp_daily = daily.groupby("product_id")["cnt"]
    feat = pd.DataFrame({
        "total_reviews":    df.groupby("product_id").size(),
        "daily_mean":       grp_daily.mean().round(4),
        "daily_std":        grp_daily.std(ddof=0).fillna(0).round(4),
        "daily_max":        grp_daily.max(),
        "days_with_reviews": grp_daily.count(),
    })
    feat["daily_cv"] = (feat["daily_std"] / feat["daily_mean"].clip(lower=1e-6)).round(4)

    # ── Span (first→last day) ─────────────────────────────────
    span = daily.groupby("product_id")["day"].agg(["min","max"])
    feat["span_days"] = ((span["max"] - span["min"]).dt.days + 1).clip(lower=1)
    feat["review_velocity"] = (feat["total_reviews"] / feat["span_days"]).round(4)

    # ── Burst days (vectorised per product) ───────────────────
    q75 = grp_daily.quantile(0.75).rename("q75")
    daily = daily.merge(q75.reset_index(), on="product_id", how="left")
    daily["burst_threshold"] = daily["q75"].clip(lower=1.5) * 2
    daily["burst_threshold"] = daily[["burst_threshold"]].clip(lower=3)["burst_threshold"]
    daily["is_burst"] = daily["cnt"] > daily["burst_threshold"]
    burst = daily.groupby("product_id")["is_burst"].sum().rename("burst_days")
    feat = feat.join(burst)
    feat["burst_days"]  = feat["burst_days"].fillna(0).astype(int)
    feat["burst_ratio"] = (feat["burst_days"] / feat["days_with_reviews"].clip(lower=1)).round(4)

    # ── Inter-day gaps (vectorised) ───────────────────────────
    daily_sorted = daily.sort_values(["product_id","day"])
    daily_sorted["prev_day"] = daily_sorted.groupby("product_id")["day"].shift(1)
    daily_sorted["gap"] = (daily_sorted["day"] - daily_sorted["prev_day"]).dt.days
    gap_stats = daily_sorted.groupby("product_id")["gap"].agg(
        inter_gap_mean="mean", inter_gap_std="std"
    ).fillna(0).round(4)
    feat = feat.join(gap_stats)

    # ── Rating stats ──────────────────────────────────────────
    rating_grp = df.groupby("product_id")["rating"]
    feat["rating_std"]   = rating_grp.std(ddof=0).fillna(0).round(4)
    feat["five_star_pct"] = (rating_grp.apply(lambda s: (s == 5).mean())).round(4)

    feat = feat.drop(columns=["span_days"], errors="ignore")
    feat.index.name = "product_id"
    return feat


FEATURE_COLS = [
    "daily_mean", "daily_std", "daily_max", "daily_cv",
    "burst_days", "burst_ratio", "inter_gap_mean", "inter_gap_std",
    "days_with_reviews", "review_velocity", "rating_std", "five_star_pct",
]


class TemporalModel:
    """Isolation Forest on product-level temporal features."""

    def __init__(self, save_path: str = SAVE_PATH):
        self.save_path   = save_path
        self.model       = None
        self.scaler      = None
        self.is_trained  = False

    # ── Train ─────────────────────────────────────────────────

    def train(
        self,
        csv_path: str = "data/processed/clean_reviews.csv",
        contamination: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        """
        Build temporal features and fit Isolation Forest.

        contamination : expected fraction of anomalous products (0.05–0.5)

        Returns
        -------
        dict with training info
        """
        df   = pd.read_csv(csv_path, parse_dates=["date"])
        feat = build_temporal_features(df)

        X = feat[FEATURE_COLS].fillna(0).values
        n_products = len(feat)

        # ── Adaptive hyper-params so large datasets don't hang ──
        if n_products > 10_000:
            n_est, max_samp = 50,  256
        elif n_products > 2_000:
            n_est, max_samp = 100, 512
        else:
            n_est, max_samp = 200, "auto"

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=n_est,
            max_samples=max_samp,
            contamination=contamination,
            random_state=SEED,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        # Raw anomaly score from IF: more negative = more anomalous
        raw_scores  = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        n_anomaly   = (predictions == -1).sum()
        anomaly_mask = predictions == -1

        # ── Feature comparison: anomalous vs normal means ────────
        feat_df = feat[FEATURE_COLS].fillna(0)
        feat_means_anomaly = feat_df[anomaly_mask].mean().round(4).to_dict()
        feat_means_normal  = feat_df[~anomaly_mask].mean().round(4).to_dict()

        metrics = {
            "n_products":     n_products,
            "n_anomalies":    int(n_anomaly),
            "anomaly_pct":    round(n_anomaly / n_products * 100, 1),
            "score_range":    (round(float(raw_scores.min()), 4),
                               round(float(raw_scores.max()), 4)),
            "n_estimators":   n_est,
            "max_samples":    max_samp,
            "feat_means_anomaly": feat_means_anomaly,
            "feat_means_normal":  feat_means_normal,
            "feature_cols":       FEATURE_COLS,
        }

        # ── Save ──────────────────────────────────────────────
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":  self.model,
            "scaler": self.scaler,
        }, self.save_path)

        if verbose:
            print(f"[TemporalModel] Products analysed : {len(feat)}")
            print(f"[TemporalModel] Anomalies detected: {n_anomaly} "
                  f"({metrics['anomaly_pct']}%)")
            print(f"[TemporalModel] Saved: {self.save_path}")

        return metrics

    # ── Predict ───────────────────────────────────────────────

    def predict(
        self,
        csv_path: str = "data/processed/clean_reviews.csv",
        save_path: str = "data/features/temporal_scores.csv",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute a temporal anomaly score (0–1) per product.
        Higher = more anomalous / bursty behaviour.

        Returns
        -------
        DataFrame with columns: product_id, temporal_score, is_anomaly
        """
        if not self.is_trained:
            self._load()

        df   = pd.read_csv(csv_path, parse_dates=["date"])
        feat = build_temporal_features(df)

        X        = feat[FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.transform(X)

        raw     = self.model.score_samples(X_scaled)    # negative: more anomalous
        # Rescale to [0, 1] where 1 = most anomalous
        norm = (raw - raw.max()) / (raw.min() - raw.max() + 1e-9)
        norm = np.clip(norm, 0, 1)

        is_anomaly = (self.model.predict(X_scaled) == -1).astype(int)

        out = pd.DataFrame({
            "product_id":     feat.index,
            "temporal_score": np.round(norm, 4),
            "is_anomaly":     is_anomaly,
        })

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)

        if verbose:
            print(f"[TemporalModel] Scored {len(out)} products")
            print(f"[TemporalModel] Anomalies: {is_anomaly.sum()}")
            print(f"[TemporalModel] Saved to : {save_path}")

        return out

    # ── Load ──────────────────────────────────────────────────

    def _load(self):
        if not Path(self.save_path).exists():
            raise FileNotFoundError(
                f"No saved model at {self.save_path}. "
                "Call TemporalModel().train() first."
            )
        data          = joblib.load(self.save_path)
        self.model    = data["model"]
        self.scaler   = data["scaler"]
        self.is_trained = True

    @classmethod
    def load(cls, save_path: str = SAVE_PATH) -> "TemporalModel":
        obj = cls(save_path=save_path)
        obj._load()
        return obj
