"""
Topic Modelling engine.
- BERTopic (per-category + global cross-category)  — primary
- sklearn LDA                                        — baseline / comparison
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_MODEL, N_TOPICS_PER_CAT, N_TOPICS_GLOBAL,
    LDA_N_TOPICS, LDA_MAX_ITER, MODELS_DIR, SEED, CATEGORIES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _build_bertopic(n_topics: int, embedding_model) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    representation = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation,
        nr_topics=n_topics,
        top_n_words=15,
        verbose=False,
        calculate_probabilities=False,
    )


def _run_lda(texts: List[str], n_topics: int) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(
        max_df=0.90,
        min_df=5,
        stop_words="english",
        max_features=10_000,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=LDA_MAX_ITER,
        learning_method="online",
        random_state=SEED,
        n_jobs=-1,
    )
    doc_topics = lda.fit_transform(dtm)
    return lda, vectorizer, doc_topics


def _get_lda_topic_words(lda: LatentDirichletAllocation, vectorizer: CountVectorizer, n_words: int = 10) -> List[List[str]]:
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        top_indices = comp.argsort()[-n_words:][::-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


# ── Main class ──────────────────────────────────────────────────────────────────

class TopicEngine:
    def __init__(self):
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.per_category_results: Dict[str, dict] = {}
        self.global_results: dict = {}

    # ── Per-category BERTopic ─────────────────────────────────────────────────

    def fit_per_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run BERTopic independently per category. Returns df with topic columns."""
        df = df.copy()
        df["bert_topic_id"]    = -1
        df["bert_topic_label"] = "Uncategorised"

        for cat in CATEGORIES:
            mask  = df["category"] == cat
            texts = df.loc[mask, "lemmatised_text"].tolist()
            log.info(f"[{cat}] Fitting BERTopic on {len(texts):,} docs...")

            embeddings = self.embedder.encode(
                texts, show_progress_bar=True, batch_size=256
            )

            model = _build_bertopic(N_TOPICS_PER_CAT, self.embedder)
            topics, _ = model.fit_transform(texts, embeddings)

            topic_info = model.get_topic_info()
            topic_labels = {
                row["Topic"]: row["Name"]
                for _, row in topic_info.iterrows()
            }

            df.loc[mask, "bert_topic_id"]    = topics
            df.loc[mask, "bert_topic_label"] = [topic_labels.get(t, "Unknown") for t in topics]

            self.per_category_results[cat] = {
                "model":       model,
                "embeddings":  embeddings,
                "topic_info":  topic_info,
                "topic_words": {
                    row["Topic"]: model.get_topic(row["Topic"])
                    for _, row in topic_info.iterrows()
                    if row["Topic"] != -1
                },
            }

            # Save model
            model_path = MODELS_DIR / f"bertopic_{cat}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            log.info(f"[{cat}] Found {len(topic_info)-1} topics. Saved -> {model_path}")

        return df

    # ── Global cross-category BERTopic ────────────────────────────────────────

    def fit_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run a single BERTopic on ALL categories combined."""
        log.info(f"Fitting GLOBAL BERTopic on {len(df):,} docs across all categories...")
        texts = df["lemmatised_text"].tolist()

        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, batch_size=256
        )

        model = _build_bertopic(N_TOPICS_GLOBAL, self.embedder)
        topics, _ = model.fit_transform(texts, embeddings)

        topic_info = model.get_topic_info()
        topic_labels = {
            row["Topic"]: row["Name"]
            for _, row in topic_info.iterrows()
        }

        df = df.copy()
        df["global_topic_id"]    = topics
        df["global_topic_label"] = [topic_labels.get(t, "Unknown") for t in topics]

        self.global_results = {
            "model":       model,
            "embeddings":  embeddings,
            "topic_info":  topic_info,
        }

        model_path = MODELS_DIR / "bertopic_global.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Global BERTopic: {len(topic_info)-1} topics found. Saved -> {model_path}")

        return df

    # ── LDA Baseline ──────────────────────────────────────────────────────────

    def fit_lda_per_category(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Run sklearn LDA per category as an interpretable baseline."""
        lda_results = {}
        for cat in CATEGORIES:
            mask  = df["category"] == cat
            texts = df.loc[mask, "lemmatised_text"].tolist()
            log.info(f"[{cat}] Fitting LDA ({LDA_N_TOPICS} topics)...")

            lda, vec, doc_topics = _run_lda(texts, LDA_N_TOPICS)
            topic_words = _get_lda_topic_words(lda, vec)

            dominant_topics = doc_topics.argmax(axis=1).tolist()

            lda_results[cat] = {
                "lda":            lda,
                "vectorizer":     vec,
                "topic_words":    topic_words,
                "doc_topics":     doc_topics,
                "dominant_topic": dominant_topics,
            }

            model_path = MODELS_DIR / f"lda_{cat}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"lda": lda, "vectorizer": vec}, f)

        return lda_results

    # ── Cross-category topic overlap ──────────────────────────────────────────

    def compute_cross_category_overlap(self) -> pd.DataFrame:
        """
        For each global topic, compute how much it's represented in each category.
        Returns a pivot table: rows=global_topics, cols=categories, values=fraction.
        """
        if not self.global_results:
            raise RuntimeError("Run fit_global() first.")
        return pd.DataFrame()   # populated in BI extractor from the enriched df


def run(df: pd.DataFrame) -> Tuple[pd.DataFrame, TopicEngine]:
    engine = TopicEngine()

    log.info("Step 1/3 — Per-category BERTopic")
    df = engine.fit_per_category(df)

    log.info("Step 2/3 — Global cross-category BERTopic")
    df = engine.fit_global(df)

    log.info("Step 3/3 — LDA baseline per category")
    engine.fit_lda_per_category(df)

    output_path = Path(str(__import__("config").PROCESSED_CSV).replace("processed", "with_topics"))
    df.to_csv(output_path, index=False)
    log.info(f"Saved enriched data -> {output_path}")

    return df, engine


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PROCESSED_CSV
    df = pd.read_csv(PROCESSED_CSV)
    run(df)
