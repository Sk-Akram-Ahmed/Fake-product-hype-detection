#!/usr/bin/env python3
"""
scripts/setup_project.py
─────────────────────────
Creates ALL required project directories, downloads NLTK data,
validates the environment, and prints a health-check summary.

Run from the project root:
    python scripts/setup_project.py
"""

import sys
import os
import subprocess
from pathlib import Path

# ── ANSI colors ──────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):   print(f"  {GREEN}[OK]{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}[!]{RESET}   {msg}")
def fail(msg): print(f"  {RED}[X]{RESET}   {msg}")


# ── Directory structure ───────────────────────────────────────
DIRECTORIES = [
    # Data layers
    "data/raw",
    "data/processed",
    "data/features",
    "data/interim",
    # Models
    "models/saved",
    "models/checkpoints",
    # Reports
    "reports/figures",
    "reports/tables",
    # Notebooks
    "notebooks",
    # Source code
    "src/data",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/visualization",
    "src/utils",
    # Paper
    "paper",
    # Tests
    "tests",
    # Logs
    "logs",
    # Config
    "config",
    # Scripts
    "scripts",
]

# ── Placeholder .gitkeep files ────────────────────────────────
GITKEEP_DIRS = [
    "data/raw", "data/processed", "data/features",
    "models/saved", "reports/figures", "logs",
]

# ── Required Python packages (spot-check) ────────────────────
REQUIRED_PACKAGES = [
    "numpy", "pandas", "sklearn", "scipy",
    "nltk", "spacy", "textblob", "vaderSentiment",
    "matplotlib", "seaborn", "plotly", "tqdm",
    "yaml", "loguru", "joblib", "shap",
]

NLTK_RESOURCES = [
    "punkt", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "omw-1.4",
]


def create_directories():
    print(f"\n{BOLD}[Step 1/4] Creating directory structure ...{RESET}")
    for d in DIRECTORIES:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        ok(f"Created: {d}/")

    for d in GITKEEP_DIRS:
        gk = Path(d) / ".gitkeep"
        gk.touch(exist_ok=True)


def check_python_version():
    print(f"\n{BOLD}[Step 2/4] Python version check ...{RESET}")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        ok(f"Python {version.major}.{version.minor}.{version.micro} - OK")
    else:
        fail(
            f"Python {version.major}.{version.minor} detected. "
            "Python >= 3.9 required."
        )
        sys.exit(1)


def check_packages():
    print(f"\n{BOLD}[Step 3/4] Checking installed packages ...{RESET}")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            ok(f"{pkg}")
        except ImportError:
            fail(f"{pkg} -- NOT FOUND")
            missing.append(pkg)

    if missing:
        warn(
            f"\n  {len(missing)} package(s) missing. "
            "Run: pip install -r requirements.txt"
        )
    else:
        ok("All core packages present.")


def download_nltk():
    print(f"\n{BOLD}[Step 4/4] Downloading NLTK resources ...{RESET}")
    try:
        import nltk
        for resource in NLTK_RESOURCES:
            try:
                nltk.download(resource, quiet=True)
                ok(f"NLTK: {resource}")
            except Exception as e:
                warn(f"NLTK {resource}: {e}")
    except ImportError:
        warn("NLTK not installed; skipping.")


def print_summary():
    print(f"\n{'='*60}")
    print(f"{BOLD}{GREEN}  PROJECT SETUP COMPLETE{RESET}")
    print(f"{'='*60}")
    print("""
  Next steps:
  ──────────────────────────────────────────────────
  1. Activate your virtual environment:
       Windows : hype_env\\Scripts\\activate
       macOS   : source hype_env/bin/activate

  2. Install all dependencies (if not yet done):
       pip install -r requirements.txt

  3. Place your raw datasets in:
       data/raw/amazon_reviews.csv
       data/raw/yelp_reviews.json

  4. Open the notebook:
       notebooks/01_project_overview.ipynb

  5. Proceed to Step 2 (Dataset Loading) when ready.
  --------------------------------------------------
""")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"{BOLD}  Multimodal Fake Product Hype Detection -- Setup{RESET}")
    print(f"{'='*60}")

    check_python_version()
    create_directories()
    check_packages()
    download_nltk()
    print_summary()
