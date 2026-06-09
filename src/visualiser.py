"""
Visualisation engine — generates all static Plotly charts saved to outputs/plots/.
The Streamlit app also calls these functions for interactive rendering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CATEGORIES, PLOTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PALETTE = px.colors.qualitative.Bold
SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}


# ── 1. Rating distribution ───────────────────────────────────────────────────────

def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["category", "rating"]).size().reset_index(name="count")
    fig = px.bar(
        counts, x="rating", y="count", color="category",
        barmode="group",
        title="Star Rating Distribution per Category",
        labels={"rating": "Star Rating", "count": "Review Count", "category": "Category"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_white", legend_title="Category")
    return fig


# ── 2. Sentiment distribution ────────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["category", "final_sentiment"]).size().reset_index(name="count")
    counts["pct"] = counts.groupby("category")["count"].transform(lambda x: x / x.sum() * 100).round(1)
    fig = px.bar(
        counts, x="category", y="pct", color="final_sentiment",
        barmode="stack",
        title="Sentiment Distribution per Category (%)",
        labels={"pct": "Percentage (%)", "category": "Category", "final_sentiment": "Sentiment"},
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
    return fig


# ── 3. Topic × Sentiment heatmap ────────────────────────────────────────────────

def plot_topic_sentiment_heatmap(priority_df: pd.DataFrame, topic_col: str, title: str = "") -> go.Figure:
    top = priority_df.head(20)
    fig = go.Figure(go.Bar(
        x=top["neg_ratio"],
        y=top[topic_col],
        orientation="h",
        marker_color=top["neg_ratio"],
        marker_colorscale="RdYlGn_r",
        text=top["neg_ratio"].apply(lambda v: f"{v:.1%}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=title or "Topic Priority by Negative Sentiment Ratio",
        xaxis_title="Negative Sentiment Ratio",
        yaxis_title="Topic",
        template="plotly_white",
        height=600,
        yaxis={"autorange": "reversed"},
    )
    return fig


# ── 4. Priority scatter ──────────────────────────────────────────────────────────

def plot_priority_scatter(priority_df: pd.DataFrame, topic_col: str, title: str = "") -> go.Figure:
    df = priority_df[priority_df[topic_col] != "-1_"].head(25)
    fig = px.scatter(
        df, x="norm_freq", y="neg_ratio",
        size="total", color="priority_score",
        text=topic_col,
        color_continuous_scale="RdYlGn_r",
        title=title or "Priority Matrix: Frequency vs. Negative Sentiment",
        labels={
            "norm_freq":       "Normalised Frequency →",
            "neg_ratio":       "Negative Sentiment Ratio →",
            "priority_score":  "Priority Score",
            "total":           "Review Count",
        },
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(template="plotly_white", height=600)
    return fig


# ── 5. Category scorecard table ──────────────────────────────────────────────────

def plot_scorecard_table(scorecard: pd.DataFrame) -> go.Figure:
    display_cols = [
        "category", "total_reviews", "avg_star_rating",
        "pct_positive", "pct_neutral", "pct_negative", "pct_implicit_neg",
        "n_topics_discovered"
    ]
    sc = scorecard[display_cols].copy()
    sc.columns = [
        "Category", "Reviews", "Avg ★",
        "% Positive", "% Neutral", "% Negative", "% Implicit Neg",
        "Topics Found"
    ]

    neg_vals = sc["% Negative"].tolist()
    cell_colors = [
        ["white"] * len(sc),
        ["white"] * len(sc),
        ["white"] * len(sc),
        ["#d5f5e3" if v > 40 else "white" for v in sc["% Positive"].tolist()],
        ["white"] * len(sc),
        ["#fadbd8" if v > 30 else "white" for v in neg_vals],
        ["#fdebd0" if v > 10 else "white" for v in sc["% Implicit Neg"].tolist()],
        ["white"] * len(sc),
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=list(sc.columns),
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[sc[c].tolist() for c in sc.columns],
            fill_color=cell_colors,
            align="center",
            font_size=11,
        ),
    ))
    fig.update_layout(title="Category Scorecard", template="plotly_white")
    return fig


# ── 6. Cross-category gap heatmap ────────────────────────────────────────────────

def plot_cross_category_gap(gap_df: pd.DataFrame) -> go.Figure:
    if gap_df.empty:
        return go.Figure()
    topic_col = gap_df.columns[0]
    cat_cols  = [c for c in gap_df.columns if c in CATEGORIES]
    matrix    = gap_df.head(15).set_index(topic_col)[cat_cols]
    fig = px.imshow(
        matrix,
        color_continuous_scale="RdYlGn_r",
        title="Cross-Category Gap: Negative Sentiment Ratio by Topic",
        labels={"x": "Category", "y": "Global Topic", "color": "Neg Ratio"},
        aspect="auto",
        zmin=0, zmax=1,
    )
    fig.update_layout(template="plotly_white", height=600)
    return fig


# ── 7. Aspect sentiment radar ────────────────────────────────────────────────────

def plot_aspect_radar(aspect_df: pd.DataFrame, category: str) -> go.Figure:
    sub = aspect_df[aspect_df["category"] == category]
    if sub.empty:
        return go.Figure()
    aspects   = sub["aspect"].tolist()
    neg_vals  = sub["neg_ratio"].tolist()
    pos_vals  = sub["pos_ratio"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pos_vals, theta=aspects, fill="toself", name="Positive", line_color="#2ecc71"))
    fig.add_trace(go.Scatterpolar(r=neg_vals, theta=aspects, fill="toself", name="Negative", line_color="#e74c3c"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Aspect Sentiment Radar — {category}",
        template="plotly_white",
    )
    return fig


# ── 8. Temporal trend line ───────────────────────────────────────────────────────

def plot_sentiment_trend(trend_df: pd.DataFrame) -> go.Figure:
    if trend_df.empty:
        return go.Figure()
    fig = px.line(
        trend_df, x="year_month_str", y="sentiment_score",
        color="category",
        title="Monthly Avg Sentiment Score per Category",
        labels={"year_month_str": "Month", "sentiment_score": "Avg Sentiment", "category": "Category"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig


# ── 9. Topic volume trend ────────────────────────────────────────────────────────

def plot_topic_volume_trend(volume_df: pd.DataFrame, topic: str, topic_col: str = "bert_topic_label") -> go.Figure:
    sub = volume_df[volume_df[topic_col] == topic]
    if sub.empty:
        return go.Figure()
    fig = px.bar(
        sub, x="year_month_str", y="count",
        title=f'Volume Over Time: "{topic}"',
        labels={"year_month_str": "Month", "count": "Review Count"},
        color_discrete_sequence=["#3498db"],
    )
    fig.update_layout(template="plotly_white")
    return fig


# ── 10. Word cloud per topic ─────────────────────────────────────────────────────

def generate_wordcloud(texts: List[str], title: str = "", save_path: Optional[Path] = None) -> plt.Figure:
    combined = " ".join(texts[:5000])
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="RdYlGn",
        max_words=100,
        collocations=False,
    ).generate(combined)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ── Save all static plots ────────────────────────────────────────────────────────

def save_all_static(df: pd.DataFrame, bi_outputs: Dict, temporal_outputs: Dict):
    log.info("Generating and saving all static plots...")

    plot_rating_distribution(df).write_html(str(PLOTS_DIR / "rating_distribution.html"))
    plot_sentiment_distribution(df).write_html(str(PLOTS_DIR / "sentiment_distribution.html"))

    if "scorecard" in bi_outputs:
        plot_scorecard_table(bi_outputs["scorecard"]).write_html(str(PLOTS_DIR / "scorecard.html"))

    if "cross_category_gap" in bi_outputs and not bi_outputs["cross_category_gap"].empty:
        plot_cross_category_gap(bi_outputs["cross_category_gap"]).write_html(str(PLOTS_DIR / "cross_category_gap.html"))

    if "aspect_sentiment" in bi_outputs:
        for cat in CATEGORIES:
            plot_aspect_radar(bi_outputs["aspect_sentiment"], cat).write_html(
                str(PLOTS_DIR / f"aspect_radar_{cat}.html")
            )

    if "category_sentiment_trend" in temporal_outputs and not temporal_outputs["category_sentiment_trend"].empty:
        plot_sentiment_trend(temporal_outputs["category_sentiment_trend"]).write_html(
            str(PLOTS_DIR / "sentiment_trend.html")
        )

    for cat in CATEGORIES:
        key = f"priority_{cat}"
        if key in bi_outputs and not bi_outputs[key].empty:
            plot_priority_scatter(bi_outputs[key], "bert_topic_label", f"Priority Matrix — {cat}").write_html(
                str(PLOTS_DIR / f"priority_{cat}.html")
            )

    log.info(f"All plots saved to {PLOTS_DIR}")
