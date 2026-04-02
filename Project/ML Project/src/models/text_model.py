"""
src/models/text_model.py  —  Step 3: TF-IDF + Logistic Regression
------------------------------------------------------------------
Trains a fake review classifier on cleaned review text.

Pipeline:
    text_clean  →  TF-IDF (10K features, 1-2 grams)
                →  Logistic Regression
                →  fake_prob  (per review, 0-1)

Usage:
    from src.models.text_model import TextModel
    model = TextModel()
    model.train("data/processed/clean_reviews.csv")
    probs = model.predict("data/processed/clean_reviews.csv")
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight


SAVE_PATH = "models/saved/text_model.pkl"
SEED = 42


class TextModel:
    """TF-IDF + Logistic Regression fake review classifier."""

    def __init__(self, save_path: str = SAVE_PATH):
        self.save_path   = save_path
        self.pipeline    = None
        self.is_trained  = False
        self.has_labels  = False

    # ── Build sklearn pipeline ───────────────────────────────

    def _build_pipeline(self):
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",   # handles class imbalance
                random_state=SEED,
            )),
        ])

    # ── Training ──────────────────────────────────────────────

    def train(
        self,
        csv_path: str = "data/processed/clean_reviews.csv",
        text_col: str = "text_clean",
        label_col: str = "is_fake",
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """
        Train the classifier.

        If labelled data (is_fake != -1) is available, do a proper
        supervised train/eval split. Otherwise fall back to
        unsupervised scoring using vocabulary heuristics.

        Returns
        -------
        dict of evaluation metrics
        """
        df = pd.read_csv(csv_path)
        labeled = df[df[label_col] >= 0].copy() if label_col in df.columns else pd.DataFrame()

        if len(labeled) >= 20:
            # ── Supervised path ──────────────────────────────
            self.has_labels = True
            X = labeled[text_col].fillna("").astype(str)
            y = labeled[label_col].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=SEED
            )

            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X_train, y_train)

            y_pred  = self.pipeline.predict(X_test)
            y_prob  = self.pipeline.predict_proba(X_test)[:, 1]
            auc     = roc_auc_score(y_test, y_prob)
            cm      = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            report  = classification_report(
                y_test, y_pred,
                target_names=["Genuine","Fake"],
                output_dict=True,
            )

            # ── Top TF-IDF words for Fake class ──────────────
            tfidf    = self.pipeline.named_steps["tfidf"]
            clf      = self.pipeline.named_steps["clf"]
            feature_names = tfidf.get_feature_names_out()
            coefs         = clf.coef_[0]
            top_fake_idx     = coefs.argsort()[-25:][::-1]
            top_genuine_idx  = coefs.argsort()[:25]
            top_fake_words    = [{"word": feature_names[i], "score": round(float(coefs[i]), 4)}
                                 for i in top_fake_idx]
            top_genuine_words = [{"word": feature_names[i], "score": round(float(coefs[i]), 4)}
                                 for i in top_genuine_idx]

            metrics = {
                "mode":              "supervised",
                "train_n":           len(X_train),
                "test_n":            len(X_test),
                "roc_auc":           round(auc, 4),
                "report":            report,
                "confusion_matrix":  cm.tolist(),
                "roc_fpr":           fpr.tolist(),
                "roc_tpr":           tpr.tolist(),
                "top_fake_words":    top_fake_words,
                "top_genuine_words": top_genuine_words,
            }

            if verbose:
                print(f"[TextModel] Mode: supervised")
                print(f"[TextModel] Train={len(X_train)}, Test={len(X_test)}")
                print(f"[TextModel] ROC-AUC: {auc:.4f}")
                print(classification_report(y_test, y_pred,
                                            target_names=["Genuine", "Fake"]))
        else:
            # ── Heuristic path (no labels) ───────────────────
            self.has_labels = False
            X = df[text_col].fillna("").astype(str)

            # "Fake" text tends to be ALL CAPS, lots of exclamation marks,
            # very short, and repetitive — we create a proxy label using
            # a simple heuristic score, then train the TF-IDF to replicate it.
            def heuristic_label(t: str) -> int:
                score = 0
                t_raw = str(t)
                if t.upper() == t and len(t) > 5:       score += 2  # all caps
                if t_raw.count("!") >= 2:                score += 1
                if len(t_raw.split()) <= 5:              score += 1
                upper_words = sum(1 for w in t_raw.split() if w.isupper() and len(w) > 2)
                if upper_words >= 2:                     score += 1
                return int(score >= 3)

            y_proxy = X.apply(heuristic_label)
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X, y_proxy)
            metrics = {"mode": "heuristic (no labels found)"}

            if verbose:
                print("[TextModel] Mode: heuristic (no is_fake labels in data)")

        # ── Save ──────────────────────────────────────────────
        self.is_trained = True
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "has_labels": self.has_labels},
                    self.save_path)

        if verbose:
            print(f"[TextModel] Saved: {self.save_path}")

        return metrics

    # ── Prediction ────────────────────────────────────────────

    def predict(
        self,
        csv_path: str = "data/processed/clean_reviews.csv",
        text_col: str = "text_clean",
        save_path: str = "data/features/text_probs.csv",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Apply trained model to all reviews and return fake probabilities.

        Returns
        -------
        DataFrame with columns: review_id, product_id, fake_prob
        """
        if not self.is_trained:
            self._load()

        df   = pd.read_csv(csv_path)
        X    = df[text_col].fillna("").astype(str)
        prob = self.pipeline.predict_proba(X)[:, 1]

        out = pd.DataFrame({
            "review_id":  df["review_id"],
            "product_id": df["product_id"],
            "fake_prob":  np.round(prob, 4),
        })

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)

        if verbose:
            print(f"[TextModel] Scored {len(out)} reviews")
            print(f"[TextModel] Mean fake_prob : {prob.mean():.4f}")
            print(f"[TextModel] Saved to       : {save_path}")

        return out

    # ── Load saved model ─────────────────────────────────────

    def _load(self):
        if not Path(self.save_path).exists():
            raise FileNotFoundError(
                f"No saved model at {self.save_path}. "
                "Call TextModel().train() first."
            )
        data = joblib.load(self.save_path)
        self.pipeline   = data["pipeline"]
        self.has_labels = data["has_labels"]
        self.is_trained = True

    @classmethod
    def load(cls, save_path: str = SAVE_PATH) -> "TextModel":
        obj = cls(save_path=save_path)
        obj._load()
        return obj
