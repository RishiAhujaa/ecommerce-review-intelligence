"""
Sentiment Analysis engine.
- VADER       : fast lexicon-based baseline (3-class: Positive / Neutral / Negative)
- DistilBERT  : binary transformer baseline (distilbert-base-uncased-finetuned-sst-2-english)
                kept as comparison column; domain-mismatched (trained on SST-2 movie reviews)
- RoBERTa     : primary model (cardiffnlp/twitter-roberta-base-sentiment-latest)
                3-class native, trained on 124M tweets — better domain fit for reviews
- Implicit    : pattern-based implicit negativity detection (from preprocessor)

Columns written:
  vader_score         : float [-1, 1]   VADER compound
  vader_label         : str             Positive / Neutral / Negative
  distilbert_label    : str             positive / negative  (baseline, binary)
  distilbert_score    : float [0,1]     DistilBERT confidence
  roberta_label       : str             positive / neutral / negative  (primary)
  roberta_score       : float [0,1]     RoBERTa confidence
  sentiment_score     : float [-1, 1]   fused numeric score
  final_sentiment     : str             Positive / Neutral / Negative  (fused label)
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SENTIMENT_MODEL, SENTIMENT_BATCH,
    VADER_COMPOUND_POS, VADER_COMPOUND_NEG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_LABEL_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

ROBERTA_MODEL    = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_BATCH    = 64   # smaller than DistilBERT due to larger model
DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


# ── VADER ─────────────────────────────────────────────────────────────────────────

def run_vader(texts: List[str]) -> pd.DataFrame:
    log.info("Running VADER sentiment...")
    analyser = SentimentIntensityAnalyzer()
    scores, labels = [], []
    for text in tqdm(texts, desc="VADER"):
        compound = analyser.polarity_scores(text)["compound"]
        scores.append(round(compound, 4))
        if compound >= VADER_COMPOUND_POS:
            labels.append("Positive")
        elif compound <= VADER_COMPOUND_NEG:
            labels.append("Negative")
        else:
            labels.append("Neutral")
    return pd.DataFrame({"vader_score": scores, "vader_label": labels})


# ── DistilBERT (baseline) ─────────────────────────────────────────────────────────

def run_distilbert(texts: List[str]) -> pd.DataFrame:
    """Binary transformer baseline — kept for comparison in the paper."""
    log.info(f"Running DistilBERT baseline on {len(texts):,} texts...")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "sentiment-analysis",
        model=DISTILBERT_MODEL,
        tokenizer=DISTILBERT_MODEL,
        device=device,
        truncation=True,
        max_length=512,
        batch_size=SENTIMENT_BATCH,
    )
    labels, scores = [], []
    for i in tqdm(range(0, len(texts), SENTIMENT_BATCH), desc="DistilBERT"):
        batch = texts[i: i + SENTIMENT_BATCH]
        for r in pipe(batch):
            raw = r["label"].lower()
            labels.append("positive" if "pos" in raw else "negative")
            scores.append(round(r["score"], 4))
    return pd.DataFrame({"distilbert_label": labels, "distilbert_score": scores})


# ── RoBERTa (primary) ─────────────────────────────────────────────────────────────

def run_roberta(texts: List[str]) -> pd.DataFrame:
    """
    Primary model: cardiffnlp/twitter-roberta-base-sentiment-latest.
    Native 3-class (positive / neutral / negative), trained on 124M tweets.
    Better domain fit for informal product review text than SST-2.
    """
    log.info(f"Running RoBERTa on {len(texts):,} texts (batch={ROBERTA_BATCH})...")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "sentiment-analysis",
        model=ROBERTA_MODEL,
        tokenizer=ROBERTA_MODEL,
        device=device,
        truncation=True,
        max_length=512,
        batch_size=ROBERTA_BATCH,
    )
    labels, scores = [], []
    for i in tqdm(range(0, len(texts), ROBERTA_BATCH), desc="RoBERTa"):
        batch = texts[i: i + ROBERTA_BATCH]
        for r in pipe(batch):
            raw = r["label"].lower()
            if "pos" in raw:
                norm = "positive"
            elif "neg" in raw:
                norm = "negative"
            else:
                norm = "neutral"
            labels.append(norm)
            scores.append(round(r["score"], 4))
    return pd.DataFrame({"roberta_label": labels, "roberta_score": scores})


# ── Fusion ────────────────────────────────────────────────────────────────────────

def fuse_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuse VADER + RoBERTa (3-class) + implicit_negative.

    Since RoBERTa natively outputs neutral, the fusion is straightforward:
      sentiment_score = 0.4 × vader_compound + 0.6 × roberta_signed_confidence
    where roberta_signed_confidence = label_map[roberta_label] × roberta_score.

    Threshold: score >= 0.10 → Positive, score <= -0.10 → Negative, else Neutral.
    Implicit negative subtracts 0.05 before thresholding.

    Using 0.4/0.6 weights: RoBERTa is the dominant signal (better model),
    VADER provides calibration and softens extreme predictions.
    """
    df = df.copy()

    roberta_signed = df["roberta_label"].map(_LABEL_MAP).fillna(0.0) * df["roberta_score"]
    score = (0.4 * df["vader_score"] + 0.6 * roberta_signed)

    score = score.copy()
    score[df["implicit_negative"]] -= 0.05
    score = score.clip(-1.0, 1.0)

    df["sentiment_score"] = score.round(4)
    df["final_sentiment"] = score.apply(
        lambda s: "Positive" if s >= 0.10 else ("Negative" if s <= -0.10 else "Neutral")
    )

    return df


# ── Main entry ────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["clean_text"].tolist()

    # 1. VADER
    vader_df = run_vader(texts)
    df = pd.concat([df.reset_index(drop=True), vader_df], axis=1)

    # 2. DistilBERT (baseline — saved for paper comparison)
    distilbert_df = run_distilbert(texts)
    df = pd.concat([df.reset_index(drop=True), distilbert_df], axis=1)

    # 3. RoBERTa (primary)
    roberta_df = run_roberta(texts)
    df = pd.concat([df.reset_index(drop=True), roberta_df], axis=1)

    # 4. Fuse
    df = fuse_sentiment(df)

    log.info("Sentiment distribution (final):")
    log.info(df["final_sentiment"].value_counts().to_string())

    out_path = Path(__file__).parent.parent / "data" / "reviews_with_sentiment.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved -> {out_path}")
    return df
