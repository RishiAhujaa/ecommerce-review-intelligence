"""
Text preprocessing pipeline using spaCy.
Handles cleaning, lemmatisation, stopword removal, and implicit-sentiment flagging.
"""

import re
import html
import logging
from typing import List

import pandas as pd
import spacy
from tqdm import tqdm

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from config import (
    SPACY_MODEL, BATCH_SIZE_SPACY, MIN_REVIEW_LENGTH,
    MAX_REVIEW_LENGTH, IMPLICIT_NEG_PATTERNS, PROCESSED_CSV, RAW_CSV
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Compile implicit-sentiment patterns once ────────────────────────────────────
_IMPLICIT_RE = [re.compile(p, re.IGNORECASE) for p in IMPLICIT_NEG_PATTERNS]

# Extra stopwords beyond spaCy defaults
_EXTRA_STOPS = {
    "product", "item", "amazon", "order", "buy", "purchase", "bought",
    "get", "use", "used", "using", "one", "also", "would", "could",
    "really", "very", "just", "like", "well", "good", "great", "bad",
    "much", "make", "thing", "stuff", "lot", "time", "way",
}


def _clean_text(text: str) -> str:
    """Remove HTML, URLs, special characters, normalise whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)                        # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)               # URLs
    text = re.sub(r"[^a-zA-Z0-9\s\'\-]", " ", text)            # keep apostrophes/hyphens
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_REVIEW_LENGTH]


def _flag_implicit_negative(text: str) -> bool:
    """Return True if the review contains implicit negativity signals."""
    return any(p.search(text) for p in _IMPLICIT_RE)


def _lemmatise(doc) -> str:
    """Lemmatise a spaCy doc, drop stopwords, punctuation, short tokens."""
    nlp_stops = doc.vocab.lookups.get_table("lexeme_norm")  # not used directly
    tokens = [
        token.lemma_.lower()
        for token in doc
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 2
            and token.lemma_.lower() not in _EXTRA_STOPS
            and token.is_alpha
        )
    ]
    return " ".join(tokens)


class Preprocessor:
    def __init__(self):
        log.info(f"Loading spaCy model: {SPACY_MODEL}")
        self.nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        self.nlp.max_length = MAX_REVIEW_LENGTH + 100

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        Returns enriched DataFrame with columns:
          clean_text, lemmatised_text, implicit_negative, text_length
        """
        log.info(f"Preprocessing {len(df):,} reviews...")
        df = df.copy()

        # 1. Basic text cleaning
        df["clean_text"] = df["text"].fillna("").apply(_clean_text)

        # 2. Filter too-short reviews
        before = len(df)
        df = df[df["clean_text"].str.len() >= MIN_REVIEW_LENGTH].reset_index(drop=True)
        log.info(f"Dropped {before - len(df):,} reviews shorter than {MIN_REVIEW_LENGTH} chars")

        # 3. Implicit negative flag (on raw clean text, before lemmatisation)
        df["implicit_negative"] = df["clean_text"].apply(_flag_implicit_negative)
        log.info(f"Implicit negative signals found in {df['implicit_negative'].sum():,} reviews")

        # 4. Lemmatisation via spaCy batched pipeline
        log.info("Running spaCy lemmatisation (batched)...")
        lemmatised = []
        texts = df["clean_text"].tolist()
        for doc in tqdm(
            self.nlp.pipe(texts, batch_size=BATCH_SIZE_SPACY),
            total=len(texts),
            desc="Lemmatising",
        ):
            lemmatised.append(_lemmatise(doc))
        df["lemmatised_text"] = lemmatised

        # 5. Text length (word count of lemmatised)
        df["text_length"] = df["lemmatised_text"].apply(lambda t: len(t.split()))

        # 6. Drop reviews where lemmatisation left too little
        before = len(df)
        df = df[df["text_length"] >= 5].reset_index(drop=True)
        log.info(f"Dropped {before - len(df):,} reviews with <5 tokens after lemmatisation")

        log.info(f"Preprocessing done. Final size: {len(df):,} reviews")
        return df


def run():
    df_raw = pd.read_csv(RAW_CSV)
    log.info(f"Loaded raw dataset: {df_raw.shape}")

    prep = Preprocessor()
    df_processed = prep.process(df_raw)

    df_processed.to_csv(PROCESSED_CSV, index=False)
    log.info(f"Saved processed data -> {PROCESSED_CSV}")
    return df_processed


if __name__ == "__main__":
    run()
