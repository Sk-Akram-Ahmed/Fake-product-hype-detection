# Multimodal Fake Product Hype Detection

> **A research-grade, end-to-end pipeline for detecting artificially inflated product popularity using review text analysis, temporal pattern mining, and reviewer behavior modeling.**

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Authentication Setup](#authentication-setup)
6. [Quick Start](#quick-start)
7. [Pipeline Steps](#pipeline-steps)
8. [Hype Score](#hype-score)
9. [Results](#results)
10. [Research Paper](#research-paper)
11. [Contributing](#contributing)
12. [License](#license)

---

## Overview

This project builds a **multimodal hype detection system** that goes beyond simple fake review detection. Rather than flagging individual reviews, the system analyzes entire products across three orthogonal signals:

| Signal | What It Measures |
|---|---|
| **Review Text** | TF-IDF, BERT embeddings, sentiment, text similarity |
| **Temporal Patterns** | Review bursts, rating drift, velocity anomalies |
| **Reviewer Behavior** | Account age, review frequency, linguistic diversity |

These signals are fused into a single, explainable **Hype Score (0–100)** with a risk label (Low / Medium / High).

---

## System Architecture

```
Raw Data (Amazon / Yelp)
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  Clean text, normalize timestamps, handle missing data
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  TF-IDF, BERT, Sentiment, Burst metrics, Reviewer stats
└────────┬────────────┘
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼                              ▼
┌──────────────┐         ┌──────────────────┐         ┌──────────────────────┐
│ Fake Review  │         │ Temporal Anomaly │         │  Reviewer Behavior   │
│ Classifier   │         │ Detector         │         │  Model               │
│ (LR / BERT)  │         │ (IsoForest/LSTM) │         │  (RF / XGBoost)      │
└──────┬───────┘         └────────┬─────────┘         └──────────┬───────────┘
       │                          │                              │
       └──────────────────────────┴──────────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  Fusion Model   │  Gradient Boosting
                        └────────┬────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │   Hype Score    │  0–100 + Risk Label + SHAP Report
                        └─────────────────┘
```

---

## Project Structure

```
ML Project/
│
├── config/
│   └── config.yaml              # Central configuration
│
├── data/
│   ├── raw/                     # Original unmodified datasets
│   ├── interim/                 # Partially processed data
│   ├── processed/               # Final cleaned datasets
│   └── features/                # Extracted feature matrices
│
├── models/
│   ├── saved/                   # Trained model artifacts (.pkl)
│   └── checkpoints/             # LSTM / BERT checkpoints
│
├── notebooks/
│   ├── 01_project_overview.ipynb
│   ├── 02_data_loading.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_fusion_system.ipynb
│   ├── 07_evaluation.ipynb
│   ├── 08_visualization.ipynb
│   └── 09_paper_writing.ipynb
│
├── src/
│   ├── data/                    # Dataset loaders
│   ├── preprocessing/           # Text cleaning, timestamp normalization
│   ├── features/                # Feature extractors
│   ├── models/                  # Model definitions and trainers
│   ├── evaluation/              # Metrics, ablation, cross-dataset
│   ├── visualization/           # Plots and dashboards
│   └── utils/                   # Config, logging, helpers
│
├── scripts/
│   └── setup_project.py         # Environment health-check
│
├── reports/
│   ├── figures/                 # Generated plots
│   └── tables/                  # Result tables (CSV)
│
├── paper/                       # LaTeX / Markdown research paper
├── tests/                       # Unit tests
│
├── requirements.txt
├── setup.py
├── setup_env.bat                # Windows environment setup
├── .env
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- 8 GB RAM minimum (16 GB recommended for BERT)
- GPU optional (BERT runs on CPU, slower)

### Windows (Recommended)

```bat
# Clone / download the project
cd "ML Project"

# Run the automated setup
setup_env.bat
```

### Manual Setup (Cross-platform)

```bash
# Create virtual environment
python -m venv hype_env

# Activate
# Windows:
hype_env\Scripts\activate
# macOS/Linux:
source hype_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLP resources
python -c "import nltk; [nltk.download(r) for r in ['punkt','stopwords','wordnet','averaged_perceptron_tagger']]"
python -m spacy download en_core_web_sm

# Run health check
python scripts/setup_project.py
```

---

## Authentication Setup

The application includes GitHub OAuth authentication for secure access. Follow these steps to set it up:

### 1. Create GitHub OAuth App

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Fill in the details:
   - **Application name**: Hype Detection System
   - **Homepage URL**: `http://localhost:8501`
   - **Authorization callback URL**: `http://localhost:8501`
4. Click "Register application"
5. Copy the **Client ID** and generate a **Client Secret**

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your credentials
```

Update `.env` with your GitHub OAuth credentials:

```env
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
GITHUB_REDIRECT_URI=http://localhost:8501
STREAMLIT_SECRET_KEY=your_secret_key_here_generate_with_python_secrets
AUTHORIZED_USERS=  # Optional: comma-separated GitHub usernames
```

### 3. Generate Secret Key

Run this Python command to generate a secure secret key:

```python
import secrets
print(secrets.token_urlsafe(32))
```

Copy the output to `STREAMLIT_SECRET_KEY` in your `.env` file.

### 4. Install Authentication Dependencies

```bash
pip install streamlit-authenticator authlib requests
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will now require GitHub authentication before access.

---

## Quick Start

```python
from src.utils import load_config, set_seed

# Load configuration
cfg = load_config("config/config.yaml")
set_seed(cfg.project.seed)

print(f"Project: {cfg.project.name}")
print(f"Raw data path: {cfg.paths.raw_data}")
```

---

## Pipeline Steps

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_project_overview.ipynb` | Project setup & environment validation |
| 2 | `02_data_loading.ipynb` | Load Amazon & Yelp datasets |
| 3 | `03_preprocessing.ipynb` | Text cleaning, normalization |
| 4 | `04_feature_engineering.ipynb` | TF-IDF, BERT, sentiment, temporal, reviewer features |
| 5 | `05_model_training.ipynb` | Train all individual models |
| 6 | `06_fusion_system.ipynb` | Fusion model + Hype Score computation |
| 7 | `07_evaluation.ipynb` | Metrics, ablation, cross-dataset tests |
| 8 | `08_visualization.ipynb` | Interactive dashboards |
| 9 | `09_paper_writing.ipynb` | Research paper generation |

---

## Hype Score

The Hype Score is computed as a weighted combination:

$$\text{HypeScore} = 100 \times \left( 0.35 \cdot P_{fake} + 0.30 \cdot A_{temporal} + 0.20 \cdot B_{reviewer} + 0.15 \cdot V_{rating} \right)$$

| Range | Risk Level |
|-------|-----------|
| 0 – 32 | 🟢 Low |
| 33 – 65 | 🟡 Medium |
| 66 – 100 | 🔴 High |

---

## Results

*(Populated after Step 7)*

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| TF-IDF + LR | — | — | — | — |
| BERT | — | — | — | — |
| Fusion (Full) | — | — | — | — |

---

## Research Paper

The paper draft is generated in `notebooks/09_paper_writing.ipynb` and saved to `paper/hype_detection_paper.md`.

---

## License

This project is for research and educational purposes.
