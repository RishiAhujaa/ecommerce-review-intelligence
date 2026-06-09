"""
Temporal analysis engine.
Detects how topics and sentiments evolve over time using review timestamps.

Outputs:
  - Topic volume over time (monthly)
  - Sentiment drift per topic
  - Emerging vs. declining topics
  - Category-level sentiment trend
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import linregress

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CATEGORIES, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert unix timestamps or string dates to datetime."""
    df = df.copy()
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        # Unix timestamp in milliseconds
        df["date"] = pd.to_datetime(ts, unit="ms", errors="coerce")
    else:
        df["date"] = pd.to_datetime(ts, errors="coerce")

    df["year_month"] = df["date"].dt.to_period("M")
    df["year"]       = df["date"].dt.year
    return df


def topic_volume_over_time(df: pd.DataFrame, topic_col: str = "bert_topic_label") -> pd.DataFrame:
    """Monthly volume of each topic. Returns long-form DataFrame."""
    df = df.dropna(subset=["year_month"])
    pivot = (
        df.groupby(["year_month", topic_col])
        .size()
        .reset_index(name="count")
    )
    pivot["year_month_str"] = pivot["year_month"].astype(str)
    return pivot.sort_values(["year_month_str", topic_col])


def sentiment_drift_per_topic(df: pd.DataFrame, topic_col: str = "bert_topic_label") -> pd.DataFrame:
    """
    For each topic, compute monthly avg sentiment_score and fit a linear trend.
    Trend slope > 0 → improving. slope < 0 → worsening.
    """
    df = df.dropna(subset=["year_month", "sentiment_score"])
    monthly = (
        df.groupby(["year_month", topic_col])["sentiment_score"]
        .mean()
        .reset_index()
    )
    monthly["year_month_str"] = monthly["year_month"].astype(str)
    monthly["period_int"]     = monthly["year_month"].apply(lambda p: p.n if hasattr(p, "n") else int(str(p).replace("-", "")))

    rows = []
    for topic, grp in monthly.groupby(topic_col):
        grp = grp.sort_values("period_int")
        if len(grp) < 3:
            continue
        slope, intercept, r, p, _ = linregress(range(len(grp)), grp["sentiment_score"])
        rows.append({
            topic_col:      topic,
            "trend_slope":  round(slope, 6),
            "r_squared":    round(r**2, 4),
            "p_value":      round(p, 4),
            "n_periods":    len(grp),
            "trend":        "Improving" if slope > 0.001 else ("Worsening" if slope < -0.001 else "Stable"),
        })
    return pd.DataFrame(rows).sort_values("trend_slope")


def emerging_vs_declining_topics(df: pd.DataFrame, topic_col: str = "bert_topic_label", window: int = 12) -> pd.DataFrame:
    """
    Compare topic volume in recent `window` months vs. earlier period.
    Emerging: recent volume >> past volume.
    Declining: recent volume << past volume.
    """
    df = df.dropna(subset=["year_month"])
    all_periods = sorted(df["year_month"].unique())
    if len(all_periods) < window * 2:
        log.warning("Not enough time periods for trend detection.")
        return pd.DataFrame()

    recent_periods = all_periods[-window:]
    earlier_periods = all_periods[:-window]

    recent = df[df["year_month"].isin(recent_periods)]
    earlier = df[df["year_month"].isin(earlier_periods)]

    recent_counts  = recent[topic_col].value_counts().rename("recent_count")
    earlier_counts = earlier[topic_col].value_counts().rename("earlier_count")

    combined = pd.concat([recent_counts, earlier_counts], axis=1).fillna(0)
    combined["growth_ratio"] = (
        (combined["recent_count"] + 1) / (combined["earlier_count"] + 1)
    ).round(3)
    combined["status"] = combined["growth_ratio"].apply(
        lambda r: "Emerging" if r > 1.5 else ("Declining" if r < 0.67 else "Stable")
    )
    return combined.sort_values("growth_ratio", ascending=False).reset_index().rename(columns={"index": topic_col})


def category_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly avg sentiment per category."""
    df = df.dropna(subset=["year_month"])
    trend = (
        df.groupby(["year_month", "category"])["sentiment_score"]
        .mean()
        .reset_index()
    )
    trend["year_month_str"] = trend["year_month"].astype(str)
    return trend.sort_values(["year_month_str", "category"])


def run(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    log.info("Running temporal analysis...")

    df = _parse_timestamps(df)
    valid_dates = df["date"].notna().sum()
    log.info(f"Valid timestamps: {valid_dates:,} / {len(df):,}")

    if valid_dates < 1000:
        log.warning("Too few valid timestamps for meaningful temporal analysis.")
        return {}

    outputs = {}

    outputs["topic_volume_over_time"]    = topic_volume_over_time(df)
    outputs["sentiment_drift_per_topic"] = sentiment_drift_per_topic(df)
    outputs["emerging_vs_declining"]     = emerging_vs_declining_topics(df)
    outputs["category_sentiment_trend"]  = category_sentiment_trend(df)

    for name, frame in outputs.items():
        if not frame.empty:
            frame.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
            log.info(f"Saved -> {REPORTS_DIR / name}.csv")

    return outputs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "reviews_final.csv")
    run(df)
