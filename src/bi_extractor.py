"""
Business Intelligence Extraction engine.

Produces:
  1. Topic × Sentiment matrix per category
  2. Priority matrix  — (complaint frequency × negative sentiment) → what to fix first
  3. Category scorecards
  4. Cross-category gap analysis
  5. Universal vs. niche topics
  6. Aspect-level BI (price, quality, delivery, packaging, design, performance)
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CATEGORIES, IMPACT_FREQ_WEIGHT, IMPACT_SENTIMENT_WEIGHT,
    TOP_N_COMPLAINTS, TOP_N_PRAISES, REPORTS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Spanish stopwords that signal non-English topic clusters
_SPANISH_TOKENS = {"que", "pero", "muy", "para", "son", "bueno", "exacto",
                   "colombia", "para", "esto", "como", "con", "una", "los"}

def _is_english_topic(label: str) -> bool:
    """Return False if the topic label contains Spanish/non-English tokens."""
    words = set(label.lower().replace("_", " ").split())
    return len(words & _SPANISH_TOKENS) == 0


# ── Aspect keyword maps ─────────────────────────────────────────────────────────
ASPECTS = {
    "Quality":      ["quality", "durable", "durability", "cheap", "well made", "flimsy", "sturdy", "solid", "broke", "break", "broke down", "material"],
    "Price/Value":  ["price", "expensive", "cheap", "value", "worth", "cost", "overpriced", "affordable", "money", "budget"],
    "Delivery":     ["delivery", "shipping", "arrived", "package", "delayed", "late", "fast", "quick", "slow", "days", "weeks", "ship"],
    "Performance":  ["performance", "works", "fast", "slow", "battery", "speed", "efficient", "power", "lag", "reliable", "accurate"],
    "Design/Look":  ["design", "look", "appearance", "color", "colour", "size", "fit", "style", "aesthetic", "beautiful", "ugly"],
    "Packaging":    ["packaging", "box", "packed", "wrapper", "unboxing", "damaged", "wrapped"],
    "Customer Service": ["service", "support", "refund", "return", "response", "helpful", "rude", "replacement"],
    "Ease of Use":  ["easy", "simple", "instructions", "setup", "install", "confusing", "intuitive", "user friendly", "complicated"],
}


# ── 1. Topic × Sentiment matrix ─────────────────────────────────────────────────

def topic_sentiment_matrix(df: pd.DataFrame, topic_col: str = "bert_topic_label") -> pd.DataFrame:
    """Returns pivot: rows=topics, cols=sentiment labels, values=count."""
    pivot = (
        df.groupby([topic_col, "final_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot[["Positive", "Neutral", "Negative"]].sum(axis=1)
    pivot["neg_ratio"]  = (pivot["Negative"] / pivot["total"]).round(3)
    pivot["pos_ratio"]  = (pivot["Positive"] / pivot["total"]).round(3)
    return pivot.sort_values("neg_ratio", ascending=False)


# ── 2. Priority (Impact) Matrix ──────────────────────────────────────────────────

def priority_matrix(df: pd.DataFrame, topic_col: str = "bert_topic_label") -> pd.DataFrame:
    """
    Priority score = freq_weight × normalised_freq + sentiment_weight × neg_ratio
    Higher score = fix this topic first.
    """
    tsm = topic_sentiment_matrix(df, topic_col)
    tsm = tsm[tsm[topic_col] != "-1_"]  # drop noise topic

    max_total = tsm["total"].max()
    tsm["norm_freq"]       = tsm["total"] / max_total
    tsm["priority_score"]  = (
        IMPACT_FREQ_WEIGHT * tsm["norm_freq"]
        + IMPACT_SENTIMENT_WEIGHT * tsm["neg_ratio"]
    ).round(4)

    return tsm.sort_values("priority_score", ascending=False).reset_index(drop=True)


# ── 3. Category Scorecard ────────────────────────────────────────────────────────

def category_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate BI metrics per category."""
    rows = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        pos = (sub["final_sentiment"] == "Positive").mean()
        neg = (sub["final_sentiment"] == "Negative").mean()
        neu = (sub["final_sentiment"] == "Neutral").mean()
        avg_rating  = sub["rating"].mean()
        avg_sent    = sub["sentiment_score"].mean()
        impl_neg    = sub["implicit_negative"].mean()
        n_topics    = sub["bert_topic_id"].nunique()
        # top_complaint = highest neg_ratio topic (min 100 reviews, English only)
        # top_praise    = highest pos_ratio topic (min 100 reviews, English only)
        tsm = topic_sentiment_matrix(sub, "bert_topic_label")
        tsm_eng = tsm[
            ~tsm["bert_topic_label"].str.startswith("-1_") &
            (tsm["total"] >= 100) &
            tsm["bert_topic_label"].apply(_is_english_topic)
        ]
        top_complaint_topic = tsm_eng.sort_values("neg_ratio", ascending=False).iloc[0]["bert_topic_label"] if not tsm_eng.empty else "N/A"
        top_praise_topic = tsm_eng.sort_values("pos_ratio", ascending=False).iloc[0]["bert_topic_label"] if not tsm_eng.empty else "N/A"
        rows.append({
            "category":            cat,
            "total_reviews":       len(sub),
            "avg_star_rating":     round(avg_rating, 2),
            "avg_sentiment_score": round(avg_sent, 3),
            "pct_positive":        round(pos * 100, 1),
            "pct_neutral":         round(neu * 100, 1),
            "pct_negative":        round(neg * 100, 1),
            "pct_implicit_neg":    round(impl_neg * 100, 1),
            "n_topics_discovered": n_topics,
            "top_complaint_topic": top_complaint_topic,
            "top_praise_topic":    top_praise_topic,
        })
    return pd.DataFrame(rows)


# ── 4. Cross-category Gap Analysis ──────────────────────────────────────────────

def cross_category_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each global topic, compute neg_ratio per category.
    Reveals: which topic hurts one category but not others.
    """
    if "global_topic_label" not in df.columns:
        log.warning("global_topic_label not found — skipping cross-category gap.")
        return pd.DataFrame()

    pivot = (
        df.groupby(["global_topic_label", "category"])
        .apply(lambda g: (g["final_sentiment"] == "Negative").mean(), include_groups=False)
        .unstack(fill_value=0)
        .round(3)
    )
    pivot["max_gap"] = pivot.max(axis=1) - pivot.min(axis=1)
    result = pivot.sort_values("max_gap", ascending=False).reset_index()
    # Drop noise and non-English topics
    result = result[~result["global_topic_label"].str.startswith("-1_")]
    result = result[result["global_topic_label"].apply(_is_english_topic)]
    return result


# ── 5. Aspect-level Sentiment ────────────────────────────────────────────────────

def aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each predefined aspect, compute avg sentiment and freq across categories.
    """
    rows = []
    for aspect, keywords in ASPECTS.items():
        pattern = "|".join(keywords)
        mask = df["clean_text"].str.contains(pattern, case=False, na=False, regex=True)
        sub = df[mask]
        if len(sub) < 10:
            continue
        for cat in CATEGORIES:
            cat_sub = sub[sub["category"] == cat]
            if len(cat_sub) < 5:
                continue
            rows.append({
                "aspect":        aspect,
                "category":      cat,
                "mention_count": len(cat_sub),
                "avg_sentiment": round(cat_sub["sentiment_score"].mean(), 3),
                "neg_ratio":     round((cat_sub["final_sentiment"] == "Negative").mean(), 3),
                "pos_ratio":     round((cat_sub["final_sentiment"] == "Positive").mean(), 3),
            })
    return pd.DataFrame(rows).sort_values(["aspect", "neg_ratio"], ascending=[True, False])


# ── 6. Top complaints & praises per category ─────────────────────────────────────

def top_complaints_praises(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Top complaints = topics with highest neg_ratio (min 30 reviews).
    Top praises    = topics with highest pos_ratio (min 30 reviews).
    Noise/non-ASCII topics are excluded.
    """
    results = {}
    for cat in CATEGORIES:
        sub  = df[df["category"] == cat]
        pmat = topic_sentiment_matrix(sub, "bert_topic_label")

        # Filter noise, non-English, and tiny topics
        pmat = pmat[~pmat["bert_topic_label"].str.startswith("-1_")]
        pmat = pmat[pmat["total"] >= 80]
        pmat = pmat[pmat["bert_topic_label"].apply(_is_english_topic)]

        neg = (
            pmat.sort_values("neg_ratio", ascending=False)
            .head(TOP_N_COMPLAINTS)[["bert_topic_label", "Negative", "neg_ratio"]]
            .rename(columns={"bert_topic_label": "topic", "Negative": "count"})
        )
        neg["type"] = "complaint"

        pos = (
            pmat.sort_values("pos_ratio", ascending=False)
            .head(TOP_N_PRAISES)[["bert_topic_label", "Positive", "pos_ratio"]]
            .rename(columns={"bert_topic_label": "topic", "Positive": "count"})
        )
        pos["type"] = "praise"

        results[cat] = pd.concat([neg, pos], ignore_index=True)
    return results


# ── Main runner ──────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    log.info("Extracting Business Intelligence...")

    outputs = {}

    # Per-category priority matrices
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        pmat = priority_matrix(sub, "bert_topic_label")
        outputs[f"priority_{cat}"] = pmat

    # Global priority
    global_pmat = priority_matrix(df, "global_topic_label") if "global_topic_label" in df.columns else pd.DataFrame()
    outputs["priority_global"] = global_pmat

    # Category scorecard
    scorecard = category_scorecard(df)
    outputs["scorecard"] = scorecard
    scorecard.to_csv(REPORTS_DIR / "category_scorecard.csv", index=False)
    log.info(f"Scorecard saved -> {REPORTS_DIR / 'category_scorecard.csv'}")

    # Cross-category gap
    gap = cross_category_gap(df)
    outputs["cross_category_gap"] = gap
    if not gap.empty:
        gap.to_csv(REPORTS_DIR / "cross_category_gap.csv", index=False)

    # Aspect sentiment
    asp = aspect_sentiment(df)
    outputs["aspect_sentiment"] = asp
    asp.to_csv(REPORTS_DIR / "aspect_sentiment.csv", index=False)

    # Top complaints & praises
    cp = top_complaints_praises(df)
    outputs["complaints_praises"] = cp
    for cat, frame in cp.items():
        frame.to_csv(REPORTS_DIR / f"complaints_praises_{cat}.csv", index=False)

    log.info("BI extraction complete.")
    return outputs


if __name__ == "__main__":
    from config import PROCESSED_CSV
    df = pd.read_csv(PROCESSED_CSV.parent / "reviews_with_sentiment.csv")
    run(df)
