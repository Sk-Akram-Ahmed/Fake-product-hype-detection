# ============================================================
# src/utils/helpers.py
# Shared utility functions used across all pipeline stages
# ============================================================

import os
import random
import json
import pickle
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd


# ── Reproducibility ─────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Directory Management ─────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ── I/O Helpers ─────────────────────────────────────────────

def save_pickle(obj: Any, path: str) -> None:
    """Serialize any Python object to a .pkl file."""
    ensure_dirs(str(Path(path).parent))
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """Deserialize a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Save a dict / list to JSON."""
    ensure_dirs(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── DataFrame Utilities ──────────────────────────────────────

def memory_usage(df: pd.DataFrame) -> str:
    """Return human-readable memory usage of a DataFrame."""
    mem = df.memory_usage(deep=True).sum()
    for unit in ["B", "KB", "MB", "GB"]:
        if mem < 1024:
            return f"{mem:.2f} {unit}"
        mem /= 1024
    return f"{mem:.2f} TB"


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast numeric columns to the smallest possible dtype to save RAM.
    Particularly useful for large review datasets.
    """
    before = memory_usage(df)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    after = memory_usage(df)
    print(f"Memory reduced: {before} → {after}")
    return df


def dataframe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table for each column:
    dtype, non-null count, null%, unique values, sample values.
    """
    rows = []
    for col in df.columns:
        rows.append({
            "column":     col,
            "dtype":      str(df[col].dtype),
            "non_null":   df[col].notna().sum(),
            "null_pct":   round(df[col].isna().mean() * 100, 2),
            "n_unique":   df[col].nunique(),
            "sample":     str(df[col].dropna().sample(
                              min(3, df[col].notna().sum()),
                              random_state=42).tolist()),
        })
    return pd.DataFrame(rows)


# ── Time Utilities ───────────────────────────────────────────

def timestamp_str() -> str:
    """Return current timestamp as a compact string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse a timestamp column to pandas datetime,
    handling common formats gracefully.
    """
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


# ── Evaluation Helpers ───────────────────────────────────────

def hype_score_to_risk(score: float) -> str:
    """
    Convert a numeric hype score (0–100) to a risk label.

    Parameters
    ----------
    score : float — hype score in [0, 100]

    Returns
    -------
    str — 'Low' | 'Medium' | 'High'
    """
    if score < 33:
        return "Low"
    elif score < 66:
        return "Medium"
    return "High"
