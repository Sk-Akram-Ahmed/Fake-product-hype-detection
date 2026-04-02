"""
app.py  —  Step 6: Streamlit Dashboard
---------------------------------------
Fake Product Hype Detection — Interactive UI

Run:
    streamlit run app.py

Features:
  - CSV file uploader (or load pre-processed results)
  - Auto-runs pipeline on upload
  - Product selector dropdown
  - Hype Score gauge chart
  - Risk level badge (green / yellow / red)
  - Review trend chart (daily review volume)
  - Rating distribution chart
  - Signal breakdown bar chart
  - Top-risk leaderboard table
  - Per-review detail table with fake probability
"""

import sys
import io
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Make src importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Import Authentication ─────────────────────────────────────
try:
    from src.auth import github_auth
    AUTH_ENABLED = True
except ImportError:
    AUTH_ENABLED = False
    st.warning("🔐 Authentication module not found. Running without authentication.")


# ════════════════════════════════════════════════════════════
# Authentication Check
# ════════════════════════════════════════════════════════════
if AUTH_ENABLED and not github_auth.check_authentication():
    github_auth.login_page()
    st.stop()


# ════════════════════════════════════════════════════════════
# Page config  (MUST be first Streamlit call)
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hype Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════
# Custom CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .risk-high   { background:#ff4b4b; color:white; padding:6px 14px;
                 border-radius:20px; font-weight:700; font-size:1.1rem; }
  .risk-medium { background:#ffa500; color:white; padding:6px 14px;
                 border-radius:20px; font-weight:700; font-size:1.1rem; }
  .risk-low    { background:#21c354; color:white; padding:6px 14px;
                 border-radius:20px; font-weight:700; font-size:1.1rem; }
  .metric-box  { background:#1e2130; border-radius:10px;
                 padding:14px 20px; text-align:center; }
  .stPlotlyChart { border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════

RISK_CSS = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
RISK_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}


def risk_badge(level: str) -> str:
    cls = RISK_CSS.get(level, "risk-low")
    return f'<span class="{cls}">{RISK_EMOJI.get(level,"")} {level} Risk</span>'


def gauge_chart(score: float, title: str = "Hype Score") -> go.Figure:
    """Plotly gauge chart for the hype score (0–100)."""
    if score < 33:
        color = "#21c354"
    elif score < 66:
        color = "#ffa500"
    else:
        color = "#ff4b4b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 22}},
        number={"font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "gray",
            "steps": [
                {"range": [0,  33], "color": "rgba(33,195,84,0.15)"},
                {"range": [33, 66], "color": "rgba(255,165,0,0.15)"},
                {"range": [66, 100],"color": "rgba(255,75,75,0.15)"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.85,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def trend_chart(reviews_df: pd.DataFrame, product_id: str) -> go.Figure:
    """Daily review count + 7-day rolling average."""
    df = reviews_df[reviews_df["product_id"] == product_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby(df["date"].dt.date).size().reset_index()
    daily.columns = ["date", "count"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling7"] = daily["count"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["count"],
        name="Daily Reviews",
        marker_color="rgba(99,110,250,0.7)",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling7"],
        name="7-day Average",
        line=dict(color="#ffa500", width=2.5),
        mode="lines",
    ))

    # Highlight burst days (top quartile)
    q75 = daily["count"].quantile(0.75)
    burst_days = daily[daily["count"] > q75 * 1.5]
    if len(burst_days):
        fig.add_trace(go.Scatter(
            x=burst_days["date"], y=burst_days["count"],
            mode="markers",
            name="Burst Days",
            marker=dict(color="#ff4b4b", size=10, symbol="x"),
        ))

    fig.update_layout(
        title=f"Review Volume Over Time — {product_id}",
        xaxis_title="Date", yaxis_title="Reviews",
        legend=dict(orientation="h", y=1.12),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color": "white"},
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def rating_chart(reviews_df: pd.DataFrame, product_id: str) -> go.Figure:
    """Star rating distribution."""
    df = reviews_df[reviews_df["product_id"] == product_id]
    counts = df["rating"].value_counts().reindex([1,2,3,4,5], fill_value=0)
    colors = ["#ff4b4b","#ff8c42","#ffd166","#06d6a0","#118ab2"]

    fig = go.Figure(go.Bar(
        x=[f"{int(r)}★" for r in counts.index],
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition="outside",
        textfont={"color":"white"},
    ))
    fig.update_layout(
        title=f"Rating Distribution — {product_id}",
        xaxis_title="Rating", yaxis_title="Count",
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color": "white"},
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def signal_chart(row: pd.Series) -> go.Figure:
    """Horizontal bar chart of individual hype signal contributions."""
    signals = {
        "Text Fake Prob":   float(row.get("text_score", 0)),
        "Temporal Anomaly": float(row.get("temporal_score", 0)),
        "5-Star %":         float(row.get("five_star_pct", 0)),
        "Burst Ratio":      float(row.get("burst_ratio", 0)),
    }
    names  = list(signals.keys())
    values = [min(v * 100, 100) for v in signals.values()]
    colors = ["#ff4b4b" if v > 60 else "#ffa500" if v > 35 else "#21c354"
              for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"color":"white"},
    ))
    fig.update_layout(
        title="Signal Breakdown",
        xaxis=dict(range=[0, 110], title="Score (0–100)"),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color": "white"},
        margin=dict(l=20, r=60, t=45, b=20),
    )
    return fig


def explanation_text(row: pd.Series) -> str:
    """Auto-generate a plain-English reason report."""
    score     = float(row["hype_score"])
    risk      = str(row["risk_level"])
    text_s    = float(row.get("text_score", 0))
    temp_s    = float(row.get("temporal_score", 0))
    five_star = float(row.get("five_star_pct", 0))
    burst     = float(row.get("burst_ratio", 0))
    daily_max = int(row.get("daily_max", 0))

    reasons = []
    if text_s > 0.45:
        reasons.append(
            f"🚩 **High language similarity** to known fake reviews "
            f"(text score: {text_s:.2f}). Fake reviews often use "
            f"superlative, repetitive language."
        )
    if temp_s > 0.55:
        reasons.append(
            f"🚩 **Unusual review burst detected** — peak of {daily_max} reviews "
            f"in a single day (temporal anomaly score: {temp_s:.2f})."
        )
    if five_star > 0.7:
        reasons.append(
            f"🚩 **{five_star*100:.0f}% of reviews are 5-star**, which is "
            f"statistically uncommon for a real product."
        )
    if burst > 0.25:
        reasons.append(
            f"🚩 **{burst*100:.0f}% of active days qualify as burst days** — "
            f"reviews are not arriving organically."
        )
    if not reasons:
        reasons.append(
            "✅ No strong hype signals detected for this product."
        )

    header = (
        f"**Hype Score: {score:.1f}/100 — {risk} Risk**\n\n"
        f"This product received a hype score of **{score:.1f}**, "
        f"indicating a **{risk.lower()} risk** of artificially inflated popularity.\n\n"
        f"**Key findings:**\n\n"
    )
    return header + "\n\n".join(reasons)


# ════════════════════════════════════════════════════════════
# Pipeline runner  — uses session_state so it only trains once
# per upload, and shows real step-by-step progress.
# ════════════════════════════════════════════════════════════

def run_full_pipeline_with_progress(csv_bytes: bytes, filename: str, cache_key: str = ""):
    """
    Run the full pipeline with live st.status progress updates.
    Results are stored in st.session_state so re-renders don't retrain.
    cache_key — caller-supplied key (filename+size) so re-runs hit cache instantly.
    """
    import hashlib
    from src.data.loader           import load_and_clean
    from src.models.text_model     import TextModel
    from src.models.temporal_model import TemporalModel
    from src.models.fusion         import compute_hype_scores

    if not cache_key:
        cache_key = f"pipeline__auto__{hashlib.md5(csv_bytes).hexdigest()}"
    metrics_key = cache_key.replace("pipeline__", "metrics__")

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    with st.status("Running pipeline...", expanded=True) as status:

        # ── Step 1: Save upload to disk ──────────────────────
        st.write("📂 Step 1/4 — Saving uploaded file...")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(csv_bytes)
            tmp_path = f.name
        file_size_mb = len(csv_bytes) / 1_048_576
        st.write(f"   File size: {file_size_mb:.1f} MB")

        # ── Step 2: Load & clean ─────────────────────────────
        st.write("🧹 Step 2/4 — Loading & cleaning data...")
        clean_df = load_and_clean(
            csv_path  = tmp_path,
            save_path = "data/processed/clean_reviews.csv",
            verbose   = False,
        )
        st.write(f"   ✅ {len(clean_df):,} reviews · "
                 f"{clean_df['product_id'].nunique()} products loaded")

        # ── Step 3: Text model ───────────────────────────────
        st.write("🤖 Step 3/4 — Training text model (TF-IDF + LR)...")
        tm = TextModel()
        metrics = tm.train(
            csv_path="data/processed/clean_reviews.csv", verbose=False
        )
        text_df = tm.predict(
            csv_path  = "data/processed/clean_reviews.csv",
            save_path = "data/features/text_probs.csv",
            verbose   = False,
        )
        auc_str = f" · ROC-AUC: {metrics['roc_auc']}" if "roc_auc" in metrics else ""
        st.write(f"   ✅ Text model trained{auc_str}")
        if "report" in metrics:
            r = metrics["report"]
            fake_p = r.get("Fake",  r.get("1", {}))
            st.write(
                f"   Fake class → Precision: {fake_p.get('precision',0):.3f} · "
                f"Recall: {fake_p.get('recall',0):.3f} · "
                f"F1: {fake_p.get('f1-score',0):.3f}"
            )

        # ── Step 4: Temporal model ───────────────────────────
        st.write("📈 Step 4/4 — Training temporal anomaly detector...")
        tp = TemporalModel()
        t_metrics = tp.train(
            csv_path="data/processed/clean_reviews.csv", verbose=False
        )
        tp.predict(
            csv_path  = "data/processed/clean_reviews.csv",
            save_path = "data/features/temporal_scores.csv",
            verbose   = False,
        )
        st.write(
            f"   ✅ {t_metrics['n_anomalies']} anomalous products "
            f"({t_metrics['anomaly_pct']}%) detected "
            f"[IF: {t_metrics['n_estimators']} trees]"
        )

        # ── Step 5: Fusion ───────────────────────────────────
        st.write("⚡ Computing Hype Scores...")
        hype_df = compute_hype_scores(
            text_probs_path       = "data/features/text_probs.csv",
            temporal_scores_path  = "data/features/temporal_scores.csv",
            clean_csv_path        = "data/processed/clean_reviews.csv",
            save_path             = "data/features/hype_scores.csv",
            verbose               = False,
        )
        high = (hype_df["risk_level"] == "High").sum()
        st.write(f"   ✅ {len(hype_df)} products scored · {high} High Risk")

        status.update(label="✅ Pipeline complete!", state="complete", expanded=False)

    clean_with_prob = clean_df.merge(
        text_df[["review_id", "fake_prob"]], on="review_id", how="left"
    )
    result = (hype_df, clean_with_prob)
    st.session_state[cache_key]                    = result
    st.session_state[metrics_key + "__text"]       = metrics
    st.session_state[metrics_key + "__temporal"]   = t_metrics
    return result


@st.cache_data(show_spinner=False)
def load_precomputed():
    """Load already-computed pipeline outputs from disk."""
    hype_df  = pd.read_csv("data/features/hype_scores.csv")
    clean_df = pd.read_csv("data/processed/clean_reviews.csv", parse_dates=["date"])
    text_df  = pd.read_csv("data/features/text_probs.csv")
    clean_with_prob = clean_df.merge(
        text_df[["review_id","fake_prob"]], on="review_id", how="left"
    )
    return hype_df, clean_with_prob


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Authentication Profile (if enabled) ───────────────────────
    if AUTH_ENABLED:
        github_auth.logout_button()
        st.divider()
    
    st.image("https://img.icons8.com/color/96/detective.png", width=64)
    st.title("Hype Detection")
    st.caption("Fake Product Hype Detection System  v1.0")
    st.divider()

    st.subheader("📂 Data Source")
    data_mode = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use pre-computed results"],
        index=1,
    )

    uploaded = None
    if data_mode == "Upload CSV file":
        uploaded = st.file_uploader(
            "Upload review CSV",
            type=["csv"],
            help=(
                "Required columns: product_id, review_text, rating, date\n"
                "Optional: reviewer_id, is_fake"
            ),
        )
        st.caption(
            "First upload triggers full pipeline (1–2 min).\n"
            "Subsequent runs use cache."
        )
    else:
        if not Path("data/features/hype_scores.csv").exists():
            st.warning(
                "No pre-computed results found.\n\n"
                "Run first:\n```\npython run_pipeline.py\n```"
            )

    st.divider()
    st.subheader("⚙️ Display Settings")
    show_reviews = st.checkbox("Show review table", value=True)
    max_reviews  = st.slider("Max reviews to show", 10, 200, 50)
    st.divider()
    st.markdown(
        "**Score Formula**\n\n"
        "```\nHype = 0.45 × text_prob\n"
        "     + 0.35 × temporal\n"
        "     + 0.10 × five_star%\n"
        "     + 0.10 × burst_ratio\n"
        "```\n\n"
        "**Risk Thresholds**\n"
        "- 🟢 Low:    0 – 32\n"
        "- 🟡 Medium: 33 – 65\n"
        "- 🔴 High:   66 – 100"
    )


# ════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════

st.title("🔍 Fake Product Hype Detection System")
st.caption(
    "Detects artificially inflated product popularity using "
    "NLP + temporal pattern analysis."
)

# ── Load data ─────────────────────────────────────────────────
hype_df           = None
reviews_df        = None
_text_metrics     = None
_temporal_metrics = None

if data_mode == "Upload CSV file" and uploaded is not None:
    # Use filename+size as cache key — safe across re-runs (no .read() needed)
    _cache_key    = f"pipeline__{uploaded.name}__{uploaded.size}"
    _metrics_key  = f"metrics__{uploaded.name}__{uploaded.size}"

    if _cache_key in st.session_state:
        # Already computed — retrieve instantly, no re-processing
        hype_df, reviews_df = st.session_state[_cache_key]
        _text_metrics     = st.session_state.get(_metrics_key + "__text")
        _temporal_metrics = st.session_state.get(_metrics_key + "__temporal")
    else:
        file_bytes = uploaded.read()
        file_mb    = len(file_bytes) / 1_048_576
        if file_mb > 150:
            st.error(
                f"File is {file_mb:.0f} MB — too large. "
                "Please use a CSV under 150 MB, or run the pipeline "
                "from the terminal: `python run_pipeline.py --data your_file.csv`"
            )
            st.stop()
        try:
            hype_df, reviews_df = run_full_pipeline_with_progress(
                file_bytes, uploaded.name, _cache_key
            )
            _text_metrics     = st.session_state.get(_metrics_key + "__text")
            _temporal_metrics = st.session_state.get(_metrics_key + "__temporal")
        except Exception as e:
            st.error(f"**Pipeline error:** {e}")
            st.exception(e)
            st.stop()

elif data_mode == "Use pre-computed results":
    _text_metrics    = None
    _temporal_metrics = None
    if Path("data/features/hype_scores.csv").exists():
        try:
            hype_df, reviews_df = load_precomputed()
        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.stop()
    else:
        st.info(
            "👈 No data loaded yet.\n\n"
            "Either:\n"
            "1. Upload a CSV using the sidebar, or\n"
            "2. Run the pipeline first:\n\n"
            "```bash\n"
            "python run_pipeline.py\n"
            "```"
        )
        st.stop()

if hype_df is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()


# ════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔎 Product Drill-Down",
    "📋 Full Leaderboard",
    "🧠 Model Metrics",
])


# ────────────────────────────────────────────────────────────
# TAB 1: Portfolio Overview
# ────────────────────────────────────────────────────────────
with tab1:
    # ── Top KPI metrics ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products",  len(hype_df))
    with col2:
        high_count = (hype_df["risk_level"] == "High").sum()
        st.metric("🔴 High Risk",    high_count,
                  delta=f"{high_count/len(hype_df)*100:.0f}%",
                  delta_color="inverse")
    with col3:
        med_count  = (hype_df["risk_level"] == "Medium").sum()
        st.metric("🟡 Medium Risk",  med_count)
    with col4:
        low_count  = (hype_df["risk_level"] == "Low").sum()
        st.metric("🟢 Low Risk",     low_count)

    st.divider()

    # ── Scatter: all products ─────────────────────────────────
    color_map = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#21c354"}
    hype_df["color"] = hype_df["risk_level"].map(color_map)

    fig_scatter = px.scatter(
        hype_df,
        x="text_score",
        y="temporal_score",
        color="risk_level",
        color_discrete_map=color_map,
        size="hype_score",
        size_max=30,
        hover_name="product_id",
        hover_data={"hype_score": True, "risk_level": False,
                    "color": False, "total_reviews": True},
        title="Risk Map: Text Score vs Temporal Score",
        labels={
            "text_score":     "Text Fake Score",
            "temporal_score": "Temporal Anomaly Score",
        },
        height=420,
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(30,33,48,0.5)",
        font={"color":"white"},
        legend_title="Risk Level",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # ── Hype score distribution ───────────────────────────
        fig_hist = px.histogram(
            hype_df, x="hype_score",
            nbins=20,
            color="risk_level",
            color_discrete_map=color_map,
            title="Hype Score Distribution",
            labels={"hype_score": "Hype Score (0–100)"},
            height=300,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(30,33,48,0.5)",
            font={"color":"white"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        # ── Risk pie chart ────────────────────────────────────
        risk_counts = hype_df["risk_level"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=[color_map[r] for r in risk_counts.index],
            hole=0.42,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            title="Risk Level Distribution",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color":"white"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Top 5 highest hype products ───────────────────────────
    st.subheader("⚠️ Top 5 Highest Hype Score Products")
    top5 = hype_df.nlargest(5, "hype_score")[
        ["product_id","hype_score","risk_level","total_reviews","daily_max"]
    ].reset_index(drop=True)
    top5.index += 1
    st.dataframe(top5, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 2: Product Drill-Down
# ────────────────────────────────────────────────────────────
with tab2:
    _sorted_hype = hype_df.sort_values("hype_score", ascending=False)
    product_list = _sorted_hype["product_id"].tolist()

    # Pre-build O(1) lookup — avoids 29K linear scans inside format_func
    _label_map = {
        row["product_id"]: f"{row['product_id']}  —  {row['hype_score']:.1f} pts  {row['risk_level']} Risk"
        for _, row in _sorted_hype[["product_id","hype_score","risk_level"]].iterrows()
    }

    selected = st.selectbox(
        "Select product to inspect:",
        product_list,
        format_func=lambda pid: _label_map.get(pid, pid),
    )

    row = hype_df[hype_df["product_id"] == selected].iloc[0]

    # ── Gauge + Risk Badge + Quick Stats ──────────────────────
    g_col, info_col = st.columns([1, 1.4])

    with g_col:
        st.plotly_chart(
            gauge_chart(float(row["hype_score"])),
            use_container_width=True,
        )
        st.markdown(
            risk_badge(str(row["risk_level"])),
            unsafe_allow_html=True,
        )

    with info_col:
        st.markdown("### Quick Stats")
        st.markdown(f"**Total Reviews:** {int(row['total_reviews'])}")
        st.markdown(f"**Peak Daily Reviews:** {int(row['daily_max'])}")
        st.markdown(f"**Avg Daily Reviews:** {float(row['daily_mean']):.1f}")
        st.markdown(f"**5-Star Rate:** {float(row['five_star_pct'])*100:.1f}%")
        st.markdown(f"**Burst Day Ratio:** {float(row['burst_ratio'])*100:.1f}%")
        st.markdown(f"**Text Fake Score:** {float(row.get('text_score',0)):.4f}")
        st.markdown(f"**Temporal Score:** {float(row.get('temporal_score',0)):.4f}")

    st.divider()

    # ── Explanation text ──────────────────────────────────────
    with st.expander("📋 Explanation Report", expanded=True):
        st.markdown(explanation_text(row))

    # ── Charts row ────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if reviews_df is not None:
            st.plotly_chart(
                trend_chart(reviews_df, selected),
                use_container_width=True,
            )
    with c2:
        if reviews_df is not None:
            st.plotly_chart(
                rating_chart(reviews_df, selected),
                use_container_width=True,
            )
            

    st.plotly_chart(signal_chart(row), use_container_width=True)

    # ── Review detail table ───────────────────────────────────
    if show_reviews and reviews_df is not None:
        st.subheader("📝 Individual Reviews")
        prod_reviews = reviews_df[
            reviews_df["product_id"] == selected
        ].copy()
        prod_reviews["date"] = pd.to_datetime(
            prod_reviews["date"], errors="coerce"
        )
        prod_reviews = prod_reviews.sort_values("fake_prob", ascending=False)

        display_cols = [c for c in
            ["review_text","rating","date","fake_prob","is_fake","reviewer_id"]
            if c in prod_reviews.columns]

        st.dataframe(
            prod_reviews[display_cols].head(max_reviews),
            use_container_width=True,
            height=320,
        )


# ────────────────────────────────────────────────────────────
# TAB 3: Full Leaderboard
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📋 All Products — Ranked by Hype Score")

    # ── Filter controls ───────────────────────────────────────
    f1, f2 = st.columns(2)
    with f1:
        risk_filter = st.multiselect(
            "Filter by risk:",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f2:
        score_range = st.slider(
            "Hype score range:",
            0, 100, (0, 100),
        )

    filtered = hype_df[
        (hype_df["risk_level"].isin(risk_filter)) &
        (hype_df["hype_score"].between(*score_range))
    ].reset_index(drop=True)
    filtered.index += 1

    display_leaderboard = filtered[[
        "product_id", "hype_score", "risk_level",
        "text_score", "temporal_score",
        "five_star_pct","burst_ratio",
        "total_reviews","daily_max",
    ]].copy()
    display_leaderboard["five_star_pct"] = (
        display_leaderboard["five_star_pct"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(display_leaderboard, use_container_width=True, height=500)

    # ── Download button ───────────────────────────────────────
    csv_download = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download Results CSV",
        data      = csv_download,
        file_name = "hype_scores_export.csv",
        mime      = "text/csv",
    )


# ────────────────────────────────────────────────────────────
# TAB 4: Model Metrics  (precision · recall · F1 · ROC · features)
# ────────────────────────────────────────────────────────────
with tab4:
    st.header("🧠 Model Performance Metrics")

    if _text_metrics is None and _temporal_metrics is None:
        st.info(
            "Upload a CSV file from the sidebar to see model metrics.\n\n"
            "Pre-computed results do not include training metrics."
        )
    else:
        # ══════════════════════════════════════════════════════
        # SECTION 1 — Text Model (TF-IDF + Logistic Regression)
        # ══════════════════════════════════════════════════════
        st.subheader("📝 Text Model — TF-IDF + Logistic Regression")

        tm = _text_metrics or {}

        if tm.get("mode") == "supervised" and "report" in tm:
            r = tm["report"]

            # ── KPI row ──────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            fake_r    = r.get("Fake",  r.get("1", {}))
            gen_r     = r.get("Genuine", r.get("0", {}))
            with k1:
                st.metric("ROC-AUC",   f"{tm['roc_auc']:.4f}")
            with k2:
                st.metric("Fake Precision", f"{fake_r.get('precision',0):.3f}")
            with k3:
                st.metric("Fake Recall",    f"{fake_r.get('recall',0):.3f}")
            with k4:
                st.metric("Fake F1-Score",  f"{fake_r.get('f1-score',0):.3f}")
            with k5:
                st.metric("Accuracy", f"{r.get('accuracy',0):.3f}")

            st.caption(f"Train samples: {tm.get('train_n',0):,}  ·  Test samples: {tm.get('test_n',0):,}")
            st.divider()

            col_roc, col_cm = st.columns(2)

            # ── ROC Curve ─────────────────────────────────────
            with col_roc:
                fpr_vals = tm.get("roc_fpr", [])
                tpr_vals = tm.get("roc_tpr", [])
                if fpr_vals and tpr_vals:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr_vals, y=tpr_vals,
                        mode="lines",
                        name=f"ROC (AUC={tm['roc_auc']:.4f})",
                        line=dict(color="#636efa", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(99,110,250,0.12)",
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0,1], y=[0,1],
                        mode="lines",
                        name="Random Baseline",
                        line=dict(color="gray", dash="dash"),
                    ))
                    fig_roc.update_layout(
                        title="ROC Curve — Fake Review Classifier",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=360,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor ="rgba(30,33,48,0.5)",
                        font={"color":"white"},
                        legend=dict(x=0.55, y=0.15),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            # ── Confusion Matrix ───────────────────────────────
            with col_cm:
                cm = tm.get("confusion_matrix")
                if cm:
                    cm_labels = ["Genuine","Fake"]
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm,
                        x=cm_labels,
                        y=cm_labels,
                        colorscale=[
                            [0,"rgba(30,33,48,0.8)"],
                            [1,"rgba(99,110,250,0.9)"],
                        ],
                        showscale=True,
                        text=[[str(v) for v in row] for row in cm],
                        texttemplate="%{text}",
                        textfont={"size":22, "color":"white"},
                    ))
                    fig_cm.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        height=360,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color":"white"},
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

            # ── Per-class report table ────────────────────────
            st.subheader("📋 Full Classification Report")
            rows_data = []
            for cls_name in ["Genuine", "Fake", "macro avg", "weighted avg"]:
                key = cls_name
                if key not in r:
                    key = {"Genuine":"0","Fake":"1"}.get(cls_name, cls_name)
                if key in r and isinstance(r[key], dict):
                    rows_data.append({
                        "Class":     cls_name,
                        "Precision": round(r[key].get("precision",0), 4),
                        "Recall":    round(r[key].get("recall",0), 4),
                        "F1-Score":  round(r[key].get("f1-score",0), 4),
                        "Support":   int(r[key].get("support",0)),
                    })

            if rows_data:
                report_df = pd.DataFrame(rows_data).set_index("Class")
                st.dataframe(report_df, use_container_width=True)

            # ── Top feature words ─────────────────────────────
            fake_words    = tm.get("top_fake_words",    [])
            genuine_words = tm.get("top_genuine_words", [])

            if fake_words or genuine_words:
                st.subheader("🔤 Top Discriminative Words (TF-IDF Coefficients)")
                wc1, wc2 = st.columns(2)

                with wc1:
                    if fake_words:
                        fw_df = pd.DataFrame(fake_words).head(20)
                        fig_fw = go.Figure(go.Bar(
                            x=fw_df["score"], y=fw_df["word"],
                            orientation="h",
                            marker_color="rgba(255,75,75,0.8)",
                            text=fw_df["score"].round(3).astype(str),
                            textposition="outside",
                            textfont={"color":"white"},
                        ))
                        fig_fw.update_layout(
                            title="Top 20 Fake-Indicating Words",
                            height=480,
                            yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor ="rgba(30,33,48,0.5)",
                            font={"color":"white"},
                            margin=dict(l=10, r=60, t=50, b=20),
                        )
                        st.plotly_chart(fig_fw, use_container_width=True)

                with wc2:
                    if genuine_words:
                        gw_df = pd.DataFrame(genuine_words).head(20)
                        gw_df["score"] = gw_df["score"].abs()
                        fig_gw = go.Figure(go.Bar(
                            x=gw_df["score"], y=gw_df["word"],
                            orientation="h",
                            marker_color="rgba(33,195,84,0.8)",
                            text=gw_df["score"].round(3).astype(str),
                            textposition="outside",
                            textfont={"color":"white"},
                        ))
                        fig_gw.update_layout(
                            title="Top 20 Genuine-Indicating Words",
                            height=480,
                            yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor ="rgba(30,33,48,0.5)",
                            font={"color":"white"},
                            margin=dict(l=10, r=60, t=50, b=20),
                        )
                        st.plotly_chart(fig_gw, use_container_width=True)

        else:
            mode = tm.get("mode", "unknown")
            st.warning(
                f"Text model ran in **{mode}** mode — "
                "no labelled `is_fake` column found in your data, "
                "so precision/recall/F1 are not available. "
                "Add an `is_fake` column (0=genuine, 1=fake) for full metrics."
            )

        # ══════════════════════════════════════════════════════
        # SECTION 2 — Temporal Model (Isolation Forest)
        # ══════════════════════════════════════════════════════
        st.divider()
        st.subheader("📈 Temporal Model — Isolation Forest")

        tmet = _temporal_metrics or {}

        if tmet:
            t1, t2, t3, t4_kpi = st.columns(4)
            with t1:
                st.metric("Products Analysed", f"{tmet.get('n_products',0):,}")
            with t2:
                st.metric("Anomalies Detected", f"{tmet.get('n_anomalies',0):,}")
            with t3:
                st.metric("Anomaly %", f"{tmet.get('anomaly_pct',0):.1f}%")
            with t4_kpi:
                st.metric("Trees Used", tmet.get("n_estimators","—"))

            st.caption(
                f"Score range: {tmet.get('score_range', ('—','—'))[0]} "
                f"to {tmet.get('score_range', ('—','—'))[1]}  "
                f"(lower = more anomalous)  ·  "
                f"Max Samples per tree: {tmet.get('max_samples','auto')}"
            )

            # ── Feature importance: anomalous vs normal mean ──
            feat_anom = tmet.get("feat_means_anomaly", {})
            feat_norm = tmet.get("feat_means_normal",  {})
            feat_cols = tmet.get("feature_cols", [])

            if feat_anom and feat_norm and feat_cols:
                st.subheader("📊 Feature Comparison: Anomalous vs Normal Products")
                feat_comp = pd.DataFrame({
                    "Feature":   feat_cols,
                    "Anomalous": [feat_anom.get(c, 0) for c in feat_cols],
                    "Normal":    [feat_norm.get(c, 0) for c in feat_cols],
                })

                fig_feat = go.Figure()
                fig_feat.add_trace(go.Bar(
                    name="Anomalous",
                    x=feat_comp["Feature"],
                    y=feat_comp["Anomalous"],
                    marker_color="rgba(255,75,75,0.85)",
                ))
                fig_feat.add_trace(go.Bar(
                    name="Normal",
                    x=feat_comp["Feature"],
                    y=feat_comp["Normal"],
                    marker_color="rgba(33,195,84,0.85)",
                ))
                fig_feat.update_layout(
                    barmode="group",
                    title="Mean Feature Values: Anomalous vs Normal Products",
                    xaxis_title="Feature",
                    yaxis_title="Mean Value",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(30,33,48,0.5)",
                    font={"color":"white"},
                    legend=dict(orientation="h", y=1.12),
                    xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(fig_feat, use_container_width=True)

                # ── Ratio chart (how many times larger for anomalous) ──
                feat_comp["Ratio"] = (
                    feat_comp["Anomalous"] / (feat_comp["Normal"] + 1e-9)
                ).round(2)
                feat_comp_sorted = feat_comp.sort_values("Ratio", ascending=False)

                fig_ratio = go.Figure(go.Bar(
                    x=feat_comp_sorted["Feature"],
                    y=feat_comp_sorted["Ratio"],
                    marker_color=[
                        "rgba(255,75,75,0.85)" if v > 1.2
                        else "rgba(255,165,0,0.85)" if v > 0.9
                        else "rgba(33,195,84,0.85)"
                        for v in feat_comp_sorted["Ratio"]
                    ],
                    text=feat_comp_sorted["Ratio"].astype(str) + "×",
                    textposition="outside",
                    textfont={"color":"white"},
                ))
                fig_ratio.update_layout(
                    title="Feature Ratio: Anomalous ÷ Normal (>1 = higher in anomalous)",
                    xaxis_title="Feature",
                    yaxis_title="Ratio",
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(30,33,48,0.5)",
                    font={"color":"white"},
                    xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(fig_ratio, use_container_width=True)

                # ── Feature table ──────────────────────────────
                st.subheader("📋 Feature Stats Table")
                st.dataframe(
                    feat_comp[["Feature","Anomalous","Normal","Ratio"]].set_index("Feature"),
                    use_container_width=True,
                )

        # ══════════════════════════════════════════════════════
        # SECTION 3 — Hype Score Distribution
        # ══════════════════════════════════════════════════════
        st.divider()
        st.subheader("📊 Hype Score Analysis")
        color_map_m = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#21c354"}

        sc1, sc2 = st.columns(2)
        with sc1:
            # Score distribution with box plot overlay
            fig_box = go.Figure()
            for risk in ["High","Medium","Low"]:
                grp = hype_df[hype_df["risk_level"] == risk]["hype_score"]
                fig_box.add_trace(go.Box(
                    y=grp,
                    name=f"{risk} ({len(grp)})",
                    marker_color=color_map_m[risk],
                    boxmean=True,
                ))
            fig_box.update_layout(
                title="Hype Score by Risk Level (Box Plot)",
                yaxis_title="Hype Score",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(30,33,48,0.5)",
                font={"color":"white"},
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with sc2:
            # Scatter: text_score vs temporal_score coloured by hype_score
            fig_density = px.scatter(
                hype_df,
                x="five_star_pct",
                y="burst_ratio",
                color="hype_score",
                color_continuous_scale="RdYlGn_r",
                size="total_reviews",
                size_max=25,
                hover_name="product_id",
                hover_data={"hype_score":True,"risk_level":True},
                title="5-Star % vs Burst Ratio (coloured by Hype Score)",
                labels={"five_star_pct":"5-Star %","burst_ratio":"Burst Ratio"},
                height=360,
            )
            fig_density.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(30,33,48,0.5)",
                font={"color":"white"},
            )
            st.plotly_chart(fig_density, use_container_width=True)

        # ── Full stats table ────────────────────────────────────
        st.subheader("📋 Complete Product Stats Table")
        numeric_cols = ["hype_score","text_score","temporal_score",
                        "five_star_pct","burst_ratio","total_reviews","daily_max"]
        display_full = hype_df.sort_values("hype_score", ascending=False)[
            ["product_id","risk_level"] + [c for c in numeric_cols if c in hype_df.columns]
        ].reset_index(drop=True)
        display_full.index += 1
        display_full["five_star_pct"] = (display_full["five_star_pct"] * 100).round(1)

        st.dataframe(display_full, use_container_width=True, height=500)

        # ── Download metrics ──────────────────────────────────
        csv_metrics = display_full.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Stats CSV",
            data=csv_metrics,
            file_name="product_stats_full.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Fake Product Hype Detection System  ·  "
    "Models: TF-IDF + LR (text) · Isolation Forest (temporal)  ·  "
    "Built with Streamlit + Plotly"
)
