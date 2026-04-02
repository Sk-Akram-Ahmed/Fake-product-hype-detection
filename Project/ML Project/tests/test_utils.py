# ============================================================
# tests/test_utils.py
# Unit tests for src/utils — run with: pytest tests/ -v
# ============================================================

import pytest
import sys
import os
from pathlib import Path

# ── Make sure src/ is on the path ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.helpers import (
    set_seed, ensure_dirs, save_pickle, load_pickle,
    save_json, load_json, memory_usage, hype_score_to_risk,
    normalize_timestamp, dataframe_summary, reduce_mem_usage,
)


class TestSetSeed:
    def test_numpy_reproducibility(self):
        import numpy as np
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        assert list(a) == list(b), "NumPy results should be identical after same seed"

    def test_different_seeds_differ(self):
        import numpy as np
        set_seed(42)
        a = np.random.rand(5)
        set_seed(99)
        b = np.random.rand(5)
        assert list(a) != list(b), "Different seeds should produce different values"


class TestHypeScoreToRisk:
    def test_low_risk(self):
        assert hype_score_to_risk(0)  == "Low"
        assert hype_score_to_risk(10) == "Low"
        assert hype_score_to_risk(32) == "Low"

    def test_medium_risk(self):
        assert hype_score_to_risk(33) == "Medium"
        assert hype_score_to_risk(50) == "Medium"
        assert hype_score_to_risk(65) == "Medium"

    def test_high_risk(self):
        assert hype_score_to_risk(66)  == "High"
        assert hype_score_to_risk(85)  == "High"
        assert hype_score_to_risk(100) == "High"


class TestPickleIO:
    def test_roundtrip(self, tmp_path):
        data = {"key": [1, 2, 3], "value": "test"}
        path = str(tmp_path / "test.pkl")
        save_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded == data

    def test_nested_objects(self, tmp_path):
        data = {"a": {"b": {"c": 42}}}
        path = str(tmp_path / "nested.pkl")
        save_pickle(data, path)
        assert load_pickle(path) == data


class TestJsonIO:
    def test_roundtrip(self, tmp_path):
        data = {"product": "P001", "score": 78.5, "risk": "High"}
        path = str(tmp_path / "report.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data


class TestMemoryUsage:
    def test_format(self):
        import pandas as pd
        import numpy as np
        df  = pd.DataFrame(np.random.rand(100, 10))
        mem = memory_usage(df)
        assert any(unit in mem for unit in ["B", "KB", "MB", "GB"])


class TestNormalizeTimestamp:
    def test_valid_dates(self):
        import pandas as pd
        s      = pd.Series(["2023-01-01", "2023-06-15", "2024-12-31"])
        result = normalize_timestamp(s)
        assert result.notna().all()
        assert result.dtype == "datetime64[ns]"

    def test_invalid_dates_become_nat(self):
        import pandas as pd
        s      = pd.Series(["not-a-date", "2023-01-01", "???"])
        result = normalize_timestamp(s)
        assert result.isna().sum() == 2


class TestDataFrameSummary:
    def test_columns(self):
        import pandas as pd
        df      = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
        summary = dataframe_summary(df)
        assert set(summary.columns) >= {"column", "dtype", "null_pct", "n_unique"}
        assert len(summary) == 2
