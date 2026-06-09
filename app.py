"""
Streamlit Dashboard — Topic Modelling & Business Intelligence
Run: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import CATEGORIES, REPORTS_DIR, PLOTS_DIR, PLOTS_DIR
from src.visualiser import (
    plot_rating_distribution,
    plot_sentiment_distribution,
    plot_topic_sentiment_heatmap,
    plot_priority_scatter,
    plot_scorecard_table,
    plot_cross_category_gap,
    plot_aspect_radar,
    plot_sentiment_trend,
    plot_topic_volume_trend,
    generate_wordcloud,
)
from src.bi_extractor import (
    topic_sentiment_matrix,
    priority_matrix,
    aspect_sentiment,
    cross_category_gap,
)
from src.agent import (
    generate_category_insight,
    generate_executive_summary,
    build_category_context,
    answer_question,
)
from src.temporal_analysis import (
    topic_volume_over_time,
    sentiment_drift_per_topic,
    emerging_vs_declining_topics,
    category_sentiment_trend,
    _parse_timestamps,
)

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Review Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin: 5px;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.9rem; margin: 0; opacity: 0.85; }
    .section-header {
        font-size: 1.5rem; font-weight: 700;
        border-left: 4px solid #764ba2; padding-left: 10px;
        margin: 20px 0 10px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────────

def _data_file_mtime() -> float:
    """Return modification time of the best available data file (used as cache key)."""
    for p in [Path("data/reviews_final.csv"), Path("data/reviews_with_sentiment.csv"),
              Path("data/reviews_with_topics.csv"), Path("data/reviews_processed.csv")]:
        if p.exists():
            return p.stat().st_mtime
    return 0.0


HF_DATASET = "RISHI010305/ecommerce-reviews-btp"
HF_FILES = [
    "reviews_with_topics.csv",
    "reviews_with_sentiment.csv",
    "reviews_final.csv",
    "reviews_processed.csv",
]

def _download_from_hf():
    """Download CSVs from Hugging Face into local data/ folder."""
    import requests
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    base_url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
    for fname in HF_FILES:
        dest = data_dir / fname
        if dest.exists():
            continue
        url = f"{base_url}/{fname}"
        st.info(f"Downloading {fname} from Hugging Face...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

@st.cache_data(show_spinner="Loading dataset...")
def load_data(_mtime: float):
    final = Path("data/reviews_final.csv")
    topics = Path("data/reviews_with_topics.csv")
    sentiment = Path("data/reviews_with_sentiment.csv")
    processed = Path("data/reviews_processed.csv")

    # If no local data found, download from Hugging Face
    if not any(p.exists() for p in [final, topics, sentiment, processed]):
        _download_from_hf()

    for p in [final, topics, sentiment, processed]:
        if p.exists():
            df = pd.read_csv(p)
            # Ensure required columns exist
            if "final_sentiment" not in df.columns:
                df["final_sentiment"] = "Neutral"
            if "sentiment_score" not in df.columns:
                df["sentiment_score"] = 0.0
            if "bert_topic_label" not in df.columns:
                df["bert_topic_label"] = "Unknown"
            if "global_topic_label" not in df.columns:
                df["global_topic_label"] = "Unknown"
            if "implicit_negative" not in df.columns:
                df["implicit_negative"] = False
            return df

    st.error("No processed data found. Please run `python pipeline.py` first.")
    st.stop()


@st.cache_data(show_spinner="Loading reports...")
def load_reports():
    reports = {}
    for f in REPORTS_DIR.glob("*.csv"):
        try:
            reports[f.stem] = pd.read_csv(f)
        except Exception:
            pass
    return reports


# ── Sidebar ───────────────────────────────────────────────────────────────────────

def sidebar(df: pd.DataFrame):
    st.sidebar.image("https://img.icons8.com/fluency/96/shopping-bag.png", width=80)
    st.sidebar.title("🛍️ Review Intelligence")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Topic Explorer", "💬 Sentiment Analysis",
         "📊 Business Intelligence", "🌍 Cross-Category Analysis",
         "⏱️ Temporal Trends", "🔎 Review Browser", "🤖 AI Insights"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")
    selected_cats = st.sidebar.multiselect(
        "Categories", options=CATEGORIES,
        default=CATEGORIES,
        format_func=lambda x: x.replace("_", " "),
    )
    rating_range = st.sidebar.slider("Star Rating", 1, 5, (1, 5))

    df_filtered = df[
        df["category"].isin(selected_cats) &
        df["rating"].between(rating_range[0], rating_range[1])
    ]

    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Reviews", f"{len(df_filtered):,}")
    st.sidebar.metric("Categories Selected", len(selected_cats))

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    return page, df_filtered


# ── Pages ─────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    st.title("🏠 E-Commerce Review Intelligence Dashboard")
    st.markdown("**Topic Modelling & Business Intelligence Extraction from 200K+ Amazon Reviews**")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Reviews", f"{len(df):,}")
    with c2:
        st.metric("Categories", df["category"].nunique())
    with c3:
        pct_pos = (df["final_sentiment"] == "Positive").mean() * 100
        st.metric("% Positive", f"{pct_pos:.1f}%")
    with c4:
        pct_neg = (df["final_sentiment"] == "Negative").mean() * 100
        st.metric("% Negative", f"{pct_neg:.1f}%")
    with c5:
        impl = df["implicit_negative"].mean() * 100
        st.metric("Implicit Neg Signals", f"{impl:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_rating_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)

    # Scorecard
    st.markdown('<div class="section-header">Category Scorecard</div>', unsafe_allow_html=True)
    scorecard_path = REPORTS_DIR / "category_scorecard.csv"
    if scorecard_path.exists():
        sc = pd.read_csv(scorecard_path)
        sc_display = sc.copy()
        sc_display["category"] = sc_display["category"].str.replace("_", " ")
        st.dataframe(
            sc_display.style
            .background_gradient(subset=["pct_positive"], cmap="Greens")
            .background_gradient(subset=["pct_negative"], cmap="Reds")
            .background_gradient(subset=["pct_implicit_neg"], cmap="Oranges")
            .format({"avg_star_rating": "{:.2f}", "avg_sentiment_score": "{:.3f}"}),
            width="stretch",
        )


def page_topic_explorer(df: pd.DataFrame):
    st.title("🔍 Topic Explorer")

    tab1, tab2 = st.tabs(["Per-Category Topics", "Global Cross-Category Topics"])

    with tab1:
        cat = st.selectbox("Select Category", CATEGORIES, format_func=lambda x: x.replace("_", " "))
        sub = df[df["category"] == cat]

        if "bert_topic_label" in sub.columns:
            topic_counts = sub["bert_topic_label"].value_counts().reset_index()
            topic_counts.columns = ["topic", "count"]
            topic_counts = topic_counts[topic_counts["topic"] != "-1_"]

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    topic_counts.head(15), x="count", y="topic",
                    orientation="h", title=f"Top Topics — {cat.replace('_',' ')}",
                    color="count", color_continuous_scale="Viridis",
                )
                fig.update_layout(yaxis={"autorange": "reversed"}, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Topics Found", topic_counts["topic"].nunique())
                st.metric("Reviews", f"{len(sub):,}")
                top_topic = topic_counts.iloc[0]["topic"] if not topic_counts.empty else "N/A"
                st.metric("Dominant Topic", top_topic[:40] + "..." if len(top_topic) > 40 else top_topic)

            # Priority matrix
            st.markdown("#### Priority Matrix (Frequency × Negative Sentiment)")
            pmat = priority_matrix(sub, "bert_topic_label")
            if not pmat.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_topic_sentiment_heatmap(pmat, "bert_topic_label", f"Negative Ratio — {cat.replace('_',' ')}"),
                        use_container_width=True
                    )
                with col2:
                    st.plotly_chart(
                        plot_priority_scatter(pmat, "bert_topic_label", f"Priority Scatter — {cat.replace('_',' ')}"),
                        use_container_width=True
                    )

            # Word cloud for selected topic
            st.markdown("#### Word Cloud for Selected Topic")
            all_topics = topic_counts["topic"].tolist()
            sel_topic  = st.selectbox("Topic", all_topics)
            topic_texts = sub[sub["bert_topic_label"] == sel_topic]["lemmatised_text"].dropna().tolist()
            if topic_texts:
                fig_wc = generate_wordcloud(topic_texts, title=sel_topic)
                st.pyplot(fig_wc)

    with tab2:
        if "global_topic_label" in df.columns:
            global_counts = df["global_topic_label"].value_counts().reset_index()
            global_counts.columns = ["topic", "count"]
            global_counts = global_counts[global_counts["topic"] != "-1_"].head(20)

            fig = px.treemap(
                global_counts, path=["topic"], values="count",
                title="Global Topic Distribution (All Categories)",
                color="count", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cross-category presence of each global topic
            st.markdown("#### Global Topic Presence per Category")
            pivot = (
                df[df["global_topic_label"] != "-1_"]
                .groupby(["global_topic_label", "category"])
                .size()
                .unstack(fill_value=0)
            )
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).round(3)
            st.dataframe(pivot_pct.style.background_gradient(cmap="YlOrRd", axis=None), width="stretch")
        else:
            st.info("Global topics not yet generated. Run the full pipeline.")


def page_sentiment(df: pd.DataFrame):
    st.title("💬 Sentiment Analysis")

    col1, col2, col3 = st.columns(3)
    for col, label, color in zip(
        [col1, col2, col3],
        ["Positive", "Neutral", "Negative"],
        ["#2ecc71", "#f39c12", "#e74c3c"]
    ):
        with col:
            pct = (df["final_sentiment"] == label).mean() * 100
            st.metric(f"% {label}", f"{pct:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Sentiment Distribution", "Model Comparison", "Implicit Sentiment"])

    with tab1:
        st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)

        # Per-category sentiment breakdown
        cat = st.selectbox("Deep-dive Category", CATEGORIES, format_func=lambda x: x.replace("_", " "), key="sent_cat")
        sub = df[df["category"] == cat]
        sent_counts = sub.groupby(["bert_topic_label", "final_sentiment"]).size().unstack(fill_value=0).reset_index()
        if not sent_counts.empty:
            sent_counts = sent_counts[sent_counts["bert_topic_label"] != "-1_"].head(15)
            fig = px.bar(
                sent_counts.melt(id_vars="bert_topic_label", var_name="Sentiment", value_name="Count"),
                x="bert_topic_label", y="Count", color="Sentiment",
                barmode="stack",
                title=f"Topic × Sentiment — {cat.replace('_',' ')}",
                color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"},
            )
            fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Comparison of all three sentiment models used in the pipeline.")
        col1, col2, col3 = st.columns(3)

        with col1:
            if "vader_label" in df.columns:
                fig = px.pie(
                    df["vader_label"].value_counts().reset_index(),
                    names="vader_label", values="count",
                    title="VADER",
                    color="vader_label",
                    color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"},
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Lexicon-based baseline.** Uses a dictionary of ~7,500 pre-scored words. Fast but can't understand context or sarcasm. Outputs Positive / Neutral / Negative.")

        with col2:
            if "distilbert_label" in df.columns:
                fig = px.pie(
                    df["distilbert_label"].value_counts().reset_index(),
                    names="distilbert_label", values="count",
                    title="DistilBERT",
                    color="distilbert_label",
                    color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Transformer baseline.** Fine-tuned on SST-2 movie reviews (Stanford). Binary only — no neutral class. Domain mismatch with product reviews causes negative bias (~66% negative).")

        with col3:
            if "roberta_label" in df.columns:
                fig = px.pie(
                    df["roberta_label"].value_counts().reset_index(),
                    names="roberta_label", values="count",
                    title="RoBERTa (Primary)",
                    color="roberta_label",
                    color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Primary model.** Trained on 124M tweets (Cardiff NLP). Native 3-class output. Better domain fit for informal product review text. Drives the final fused sentiment score.")

        st.markdown("---")
        if "vader_label" in df.columns and "roberta_label" in df.columns:
            a1 = (df["vader_label"].str.lower() == df["roberta_label"]).mean() * 100
            c1, c2 = st.columns(2)
            c1.metric("VADER ↔ RoBERTa Agreement", f"{a1:.1f}%")
            if "distilbert_label" in df.columns:
                a2 = (df["distilbert_label"] == df["roberta_label"]).mean() * 100
                c2.metric("DistilBERT ↔ RoBERTa Agreement", f"{a2:.1f}%")

    with tab3:
        st.markdown("**Implicit negativity** = review doesn't use explicit negative words, but signals dissatisfaction through context, hedging, or passive voice.")
        impl_df = df[df["implicit_negative"] == True]
        st.metric("Reviews with Implicit Negative Signals", f"{len(impl_df):,}")

        if not impl_df.empty:
            impl_by_cat = impl_df.groupby("category").size().reset_index(name="count")
            impl_by_cat["category"] = impl_by_cat["category"].str.replace("_", " ")
            fig = px.bar(impl_by_cat, x="category", y="count", title="Implicit Negative Reviews by Category",
                         color="count", color_continuous_scale="Oranges")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Sample Implicit Negative Reviews:**")
            samples = impl_df[["category", "rating", "text", "final_sentiment"]].sample(min(10, len(impl_df)))
            st.dataframe(samples, width="stretch")


def page_bi(df: pd.DataFrame):
    st.title("📊 Business Intelligence")

    tab1, tab2, tab3 = st.tabs(["Priority Matrix", "Aspect Analysis", "Top Complaints & Praises"])

    with tab1:
        cat = st.selectbox("Category", CATEGORIES, format_func=lambda x: x.replace("_", " "), key="bi_cat")
        sub = df[df["category"] == cat]
        pmat = priority_matrix(sub, "bert_topic_label")
        if not pmat.empty:
            st.markdown("""
            **Priority Score** = 0.4 × Normalised Frequency + 0.6 × Negative Sentiment Ratio

            Higher priority = more frequent AND more negatively perceived. **Fix these first.**
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_priority_scatter(pmat, "bert_topic_label"), use_container_width=True)
            with col2:
                st.plotly_chart(plot_topic_sentiment_heatmap(pmat, "bert_topic_label"), use_container_width=True)

            st.markdown("**Full Priority Table:**")
            st.dataframe(
                pmat[["bert_topic_label", "total", "neg_ratio", "pos_ratio", "priority_score"]]
                .rename(columns={"bert_topic_label": "Topic"})
                .style.background_gradient(subset=["priority_score"], cmap="RdYlGn_r")
                .background_gradient(subset=["neg_ratio"], cmap="Reds")
                .format({"neg_ratio": "{:.1%}", "pos_ratio": "{:.1%}", "priority_score": "{:.3f}"}),
                width="stretch",
            )

    with tab2:
        asp_path = REPORTS_DIR / "aspect_sentiment.csv"
        if asp_path.exists():
            asp_df = pd.read_csv(asp_path)
            cat = st.selectbox("Category", CATEGORIES, format_func=lambda x: x.replace("_", " "), key="asp_cat")
            st.plotly_chart(plot_aspect_radar(asp_df, cat), use_container_width=True)

            sub_asp = asp_df[asp_df["category"] == cat].sort_values("neg_ratio", ascending=False)
            st.dataframe(
                sub_asp[["aspect", "mention_count", "avg_sentiment", "neg_ratio", "pos_ratio"]]
                .style.background_gradient(subset=["neg_ratio"], cmap="Reds")
                .background_gradient(subset=["pos_ratio"], cmap="Greens")
                .format({"avg_sentiment": "{:.3f}", "neg_ratio": "{:.1%}", "pos_ratio": "{:.1%}"}),
                width="stretch",
            )
        else:
            st.info("Run the full pipeline to generate aspect analysis.")

    with tab3:
        cat = st.selectbox("Category", CATEGORIES, format_func=lambda x: x.replace("_", " "), key="cp_cat")
        cp_path = REPORTS_DIR / f"complaints_praises_{cat}.csv"
        if cp_path.exists():
            cp_df = pd.read_csv(cp_path)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔴 Top Complaints")
                complaints = cp_df[cp_df["type"] == "complaint"]
                fig = px.bar(complaints, x="count", y="topic", orientation="h",
                             color_discrete_sequence=["#e74c3c"])
                fig.update_layout(yaxis={"autorange": "reversed"}, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### 🟢 Top Praises")
                praises = cp_df[cp_df["type"] == "praise"]
                fig = px.bar(praises, x="count", y="topic", orientation="h",
                             color_discrete_sequence=["#2ecc71"])
                fig.update_layout(yaxis={"autorange": "reversed"}, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the full pipeline to generate complaints/praises.")


def page_cross_category(df: pd.DataFrame):
    st.title("🌍 Cross-Category Analysis")

    tab1, tab2 = st.tabs(["Gap Analysis", "Universal vs. Niche Topics"])

    with tab1:
        gap_path = REPORTS_DIR / "cross_category_gap.csv"
        if gap_path.exists():
            gap_df = pd.read_csv(gap_path)
            st.markdown("""
            **Gap Analysis**: For each global topic, how differently does negative sentiment manifest across categories?
            High max_gap = topic affects some categories far more than others.
            """)
            st.plotly_chart(plot_cross_category_gap(gap_df), use_container_width=True)
            st.dataframe(gap_df.head(20).style.background_gradient(cmap="RdYlGn_r", axis=None), width="stretch")
        else:
            st.info("Run the full pipeline to generate cross-category gap analysis.")

    with tab2:
        if "global_topic_label" in df.columns:
            cat_presence = (
                df[df["global_topic_label"] != "-1_"]
                .groupby("global_topic_label")["category"]
                .nunique()
                .reset_index()
            )
            cat_presence.columns = ["topic", "n_categories"]
            universal = cat_presence[cat_presence["n_categories"] == df["category"].nunique()]
            niche     = cat_presence[cat_presence["n_categories"] == 1]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 🌐 Universal Topics ({len(universal)})")
                st.markdown("*Appear across ALL categories*")
                st.dataframe(universal, width="stretch")
            with col2:
                st.markdown(f"#### 🎯 Niche Topics ({len(niche)})")
                st.markdown("*Specific to a single category*")
                st.dataframe(niche, width="stretch")

            fig = px.histogram(
                cat_presence, x="n_categories",
                title="Topic Coverage: How Many Categories Does Each Topic Appear In?",
                labels={"n_categories": "Number of Categories", "count": "Number of Topics"},
                color_discrete_sequence=["#764ba2"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the full pipeline to generate global topics.")


def page_temporal(df: pd.DataFrame):
    st.title("⏱️ Temporal Trends")

    df_t = _parse_timestamps(df)
    valid = df_t["date"].notna().sum()

    if valid < 100:
        st.warning(f"Only {valid} reviews have valid timestamps. Temporal analysis is limited.")
        return

    st.metric("Reviews with Valid Timestamps", f"{valid:,}")

    tab1, tab2, tab3 = st.tabs(["Sentiment Trend", "Topic Drift", "Emerging / Declining"])

    with tab1:
        trend_df = category_sentiment_trend(df_t)
        if not trend_df.empty:
            st.plotly_chart(plot_sentiment_trend(trend_df), use_container_width=True)

    with tab2:
        drift_df = sentiment_drift_per_topic(df_t)
        if not drift_df.empty:
            topic_col = drift_df.columns[0]
            st.markdown("**Trend Slope**: Positive = sentiment improving over time. Negative = worsening.")
            col1, col2, col3 = st.columns(3)
            with col1:
                improving = (drift_df["trend"] == "Improving").sum()
                st.metric("Improving Topics", improving, delta="↑ sentiment")
            with col2:
                stable = (drift_df["trend"] == "Stable").sum()
                st.metric("Stable Topics", stable)
            with col3:
                worsening = (drift_df["trend"] == "Worsening").sum()
                st.metric("Worsening Topics", worsening, delta="↓ sentiment", delta_color="inverse")

            st.dataframe(
                drift_df.style.background_gradient(subset=["trend_slope"], cmap="RdYlGn"),
                width="stretch",
            )

    with tab3:
        emerg_df = emerging_vs_declining_topics(df_t)
        if not emerg_df.empty:
            topic_col = emerg_df.columns[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Emerging Topics")
                st.dataframe(emerg_df[emerg_df["status"] == "Emerging"][[topic_col, "growth_ratio"]].head(10), width="stretch")
            with col2:
                st.markdown("#### 📉 Declining Topics")
                st.dataframe(emerg_df[emerg_df["status"] == "Declining"][[topic_col, "growth_ratio"]].head(10), width="stretch")


def page_review_browser(df: pd.DataFrame):
    st.title("🔎 Review Browser")

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter  = st.selectbox("Category", ["All"] + CATEGORIES, format_func=lambda x: x.replace("_", " "))
    with col2:
        sent_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
    with col3:
        search_term = st.text_input("Search in reviews", "")

    filtered = df.copy()
    if cat_filter != "All":
        filtered = filtered[filtered["category"] == cat_filter]
    if sent_filter != "All":
        filtered = filtered[filtered["final_sentiment"] == sent_filter]
    if search_term:
        filtered = filtered[filtered["text"].str.contains(search_term, case=False, na=False)]

    total = len(filtered)
    st.metric("Matching Reviews", f"{total:,}")

    display_cols = ["category", "rating", "final_sentiment", "sentiment_score",
                    "bert_topic_label", "implicit_negative", "text"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    PAGE_SIZE = 500
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    col_info, col_pager = st.columns([3, 2])
    with col_info:
        st.caption(f"{total:,} reviews — {total_pages} page(s) of {PAGE_SIZE}")
    with col_pager:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * PAGE_SIZE
    end   = start + PAGE_SIZE
    page_df = filtered[display_cols].iloc[start:end].rename(columns={
        "bert_topic_label":  "Topic",
        "final_sentiment":   "Sentiment",
        "sentiment_score":   "Score",
        "implicit_negative": "Implicit Neg",
    })

    page_df.index = range(start + 1, start + 1 + len(page_df))
    html = page_df.to_html(index=True, escape=True)
    st.markdown(
        f"""
        <div style="overflow-x: auto; overflow-y: auto; max-height: 520px; border: 1px solid #e0e0e0; border-radius: 6px;">
        <style>
            .review-table {{ border-collapse: collapse; white-space: nowrap; font-size: 13px; width: max-content; }}
            .review-table th {{ background: #f5f5f5; padding: 8px 12px; border-bottom: 2px solid #ddd; text-align: left; position: sticky; top: 0; z-index: 1; }}
            .review-table td {{ padding: 6px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
            .review-table tr:hover td {{ background: #fafafa; }}
        </style>
        {html.replace('<table ', '<table class="review-table" ')}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── AI Insights page ──────────────────────────────────────────────────────────────

def page_ai_insights(df: pd.DataFrame):
    st.title("🤖 AI Insights")
    st.markdown("**Groq-powered analysis** — auto-generated reports *and* a chat interface to ask anything about your data.")
    st.markdown("---")

    # API key
    import os
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.info("Enter your Groq API key to continue. Get one free at [console.groq.com](https://console.groq.com).")
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if not api_key:
        st.stop()

    # BI outputs (cached)
    @st.cache_data(show_spinner="Building BI context...")
    def _get_bi_outputs(_mtime: float):
        from src.bi_extractor import run as run_bi
        return run_bi(df)

    bi_outputs = _get_bi_outputs(_mtime=_data_file_mtime())

    # Build all contexts once (used by both tabs)
    all_contexts = []
    for cat in CATEGORIES:
        try:
            all_contexts.append(build_category_context(df, bi_outputs, cat))
        except Exception:
            pass

    # ── Two tabs ──────────────────────────────────────────────────────────────
    tab_reports, tab_chat = st.tabs(["📋 Generated Reports", "💬 Ask a Question"])

    # ── Tab 1: Generated Reports ──────────────────────────────────────────────
    with tab_reports:
        col1, col2 = st.columns([3, 1])
        with col1:
            view = st.selectbox(
                "Select view",
                ["📋 Executive Summary"] + [f"📁 {c.replace('_', ' ')}" for c in CATEGORIES],
                key="report_view",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            force = st.button("🔄 Regenerate", help="Call Groq again (ignores cache)")

        st.markdown("---")

        if view == "📋 Executive Summary":
            with st.spinner("Generating executive summary..."):
                try:
                    result = generate_executive_summary(all_contexts, api_key, force=force)
                except Exception as e:
                    st.error(f"Groq error: {e}")
                    st.stop()

            st.markdown(result["insight_text"])
            st.caption(f"Generated: {result['generated_at'][:19].replace('T', ' ')}")

            st.markdown("---")
            st.markdown("**Category Snapshot**")
            rows = []
            for ctx in all_contexts:
                rows.append({
                    "Category":      ctx["category"],
                    "Reviews":       ctx["total_reviews"],
                    "★ Avg":         ctx["avg_star_rating"],
                    "% Positive":    ctx["pct_positive"],
                    "% Neutral":     ctx["pct_neutral"],
                    "% Negative":    ctx["pct_negative"],
                    "Top Complaint": ctx["top_complaints"][0]["topic"] if ctx["top_complaints"] else "N/A",
                })
            if rows:
                snap_df = pd.DataFrame(rows)
                st.dataframe(
                    snap_df.style
                    .background_gradient(subset=["% Positive"], cmap="Greens")
                    .background_gradient(subset=["% Negative"], cmap="Reds")
                    .format({"★ Avg": "{:.2f}", "% Positive": "{:.1f}", "% Neutral": "{:.1f}", "% Negative": "{:.1f}"}),
                    width="stretch",
                )

        else:
            cat = next((c for c in CATEGORIES if c.replace("_", " ") == view.replace("📁 ", "")), CATEGORIES[0])

            with st.spinner(f"Generating insight for {cat.replace('_', ' ')}..."):
                try:
                    result = generate_category_insight(df, bi_outputs, cat, api_key, force=force)
                except Exception as e:
                    st.error(f"Groq error: {e}")
                    st.stop()

            st.markdown(result["insight_text"])
            st.caption(f"Generated: {result['generated_at'][:19].replace('T', ' ')}")

            with st.expander("📊 Data fed to Groq"):
                ctx = result.get("context", {})
                if ctx:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Reviews", f"{ctx.get('total_reviews', 0):,}")
                    c2.metric("% Positive",    f"{ctx.get('pct_positive', 0):.1f}%")
                    c3.metric("% Negative",    f"{ctx.get('pct_negative', 0):.1f}%")
                    c4.metric("Avg Rating",    f"★{ctx.get('avg_star_rating', 0):.2f}")
                    if ctx.get("top_priority_topics"):
                        st.markdown("**Priority Topics**")
                        st.dataframe(
                            pd.DataFrame(ctx["top_priority_topics"]).style
                            .background_gradient(subset=["priority_score"], cmap="RdYlGn_r")
                            .format({"neg_ratio": "{:.1f}%", "pos_ratio": "{:.1f}%", "priority_score": "{:.3f}"}),
                            width="stretch",
                        )
                    if ctx.get("aspect_sentiment"):
                        st.markdown("**Aspect Sentiment**")
                        st.dataframe(
                            pd.DataFrame(ctx["aspect_sentiment"]).style
                            .background_gradient(subset=["neg_ratio"], cmap="Reds")
                            .format({"neg_ratio": "{:.1f}%", "pos_ratio": "{:.1f}%", "avg_sentiment": "{:.3f}"}),
                            width="stretch",
                        )

    # ── Tab 2: Chat ───────────────────────────────────────────────────────────
    with tab_chat:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Clear button top-right
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        # Empty state — show hints only when no messages yet
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='text-align:center; color:#888; padding: 60px 0 20px 0;'>"
                "<h3 style='color:#555'>💬 Ask anything about your review data</h3>"
                "<p>The model has all BI numbers, topics, and sentiment scores loaded as context.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            for q, col in [
                ("Which category has the worst sentiment?", c1),
                ("What are the top complaints in Electronics?", c2),
                ("Compare delivery across all categories", c1),
                ("What should Clothing fix first?", c2),
            ]:
                with col:
                    if st.button(q, use_container_width=True):
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        st.rerun()

        # Render conversation
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])

        # Trigger response for any unanswered user message
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = answer_question(
                            all_contexts=all_contexts,
                            question=st.session_state.chat_history[-1]["content"],
                            history=st.session_state.chat_history[:-1],
                            api_key=api_key,
                        )
                    except Exception as e:
                        reply = f"⚠️ Error: {e}"
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "model", "content": reply})

        # Input pinned to bottom
        user_input = st.chat_input("Ask about your review data...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    df      = load_data(_mtime=_data_file_mtime())
    reports = load_reports()

    page, df_filtered = sidebar(df)

    if page == "🏠 Overview":
        page_overview(df_filtered)
    elif page == "🔍 Topic Explorer":
        page_topic_explorer(df_filtered)
    elif page == "💬 Sentiment Analysis":
        page_sentiment(df_filtered)
    elif page == "📊 Business Intelligence":
        page_bi(df_filtered)
    elif page == "🌍 Cross-Category Analysis":
        page_cross_category(df_filtered)
    elif page == "⏱️ Temporal Trends":
        page_temporal(df_filtered)
    elif page == "🔎 Review Browser":
        page_review_browser(df_filtered)
    elif page == "🤖 AI Insights":
        page_ai_insights(df)


if __name__ == "__main__":
    main()