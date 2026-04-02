# ============================================================
# src/utils/config_loader.py
# Loads YAML configuration and returns a plain dict or
# an attribute-accessible namespace (dot notation).
# ============================================================

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str = "config/config.yaml", 
                as_namespace: bool = True):
    """
    Load the project YAML config file.

    Parameters
    ----------
    config_path  : str  — path to config.yaml
    as_namespace : bool — if True, return dot-accessible SimpleNamespace
                          if False, return raw dict

    Returns
    -------
    SimpleNamespace | dict

    Example
    -------
    >>> cfg = load_config()
    >>> print(cfg.project.name)
    >>> print(cfg.paths.raw_data)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {path.resolve()}\n"
            "Make sure you are running from the project root directory."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw) if as_namespace else raw
