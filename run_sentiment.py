"""
Re-runs sentiment analysis:
  - Renames existing DistilBERT columns (previously mislabelled as 'roberta_*')
  - Runs cardiffnlp/twitter-roberta-base-sentiment-latest as the NEW primary model
  - Fuses VADER + RoBERTa + implicit_negative
  - Re-runs BI, temporal analysis, visualisations

Skips DistilBERT inference (already done — columns renamed from roberta_* to distilbert_*).
Estimated runtime: ~3.5 hours (RoBERTa only). Fully offline — no internet needed.

Run: python run_sentiment.py
"""

import time
import logging
import pandas as pd
from pathlib import Path
from rich.console import Console

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = Console()
ROOT = Path(__file__).parent

from src.sentiment import run_vader, run_roberta, fuse_sentiment
from src.bi_extractor import run as run_bi
from src.temporal_analysis import run as run_temporal
from src.visualiser import save_all_static


def main():
    start = time.time()
    console.rule("[bold]Sentiment Re-run (RoBERTa upgrade)[/bold]")

    # ── Load topics-enriched file ──────────────────────────────────────────
    topics_path = ROOT / "data" / "reviews_with_topics.csv"
    console.print(f"Loading {topics_path.name}...")
    df = pd.read_csv(topics_path)
    console.print(f"Loaded {len(df):,} reviews")

    # ── Rename existing DistilBERT columns (were mislabelled as roberta_*) ─
    console.print("\n[yellow]Renaming DistilBERT columns (roberta_* → distilbert_*)...[/yellow]")
    rename_map = {}
    if "roberta_label" in df.columns:
        rename_map["roberta_label"] = "distilbert_label"
    if "roberta_score" in df.columns:
        rename_map["roberta_score"] = "distilbert_score"
    if rename_map:
        df = df.rename(columns=rename_map)
        console.print(f"  Renamed: {rename_map}")

    # Drop old fused columns (will recompute)
    for col in ["sentiment_score", "final_sentiment"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    texts = df["clean_text"].tolist()

    # ── VADER (fast, ~35 sec) ──────────────────────────────────────────────
    console.rule("VADER")
    # Drop old vader columns if they exist (recompute fresh)
    for col in ["vader_score", "vader_label"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    vader_df = run_vader(texts)
    df = pd.concat([df.reset_index(drop=True), vader_df], axis=1)

    # ── RoBERTa primary (~3.5 hrs) ─────────────────────────────────────────
    console.rule("RoBERTa — cardiffnlp/twitter-roberta-base-sentiment-latest")
    console.print("[dim]Fully offline — cached model, no internet needed[/dim]")
    roberta_df = run_roberta(texts)
    df = pd.concat([df.reset_index(drop=True), roberta_df], axis=1)

    # ── Fuse ──────────────────────────────────────────────────────────────
    console.rule("Fusing")
    df = fuse_sentiment(df)

    console.print("\n[bold]Sentiment distribution:[/bold]")
    console.print(df["final_sentiment"].value_counts().to_string())
    console.print(f"\nDistilBERT baseline (for paper comparison): {df['distilbert_label'].value_counts().to_dict()}")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(ROOT / "data" / "reviews_with_sentiment.csv", index=False)
    df.to_csv(ROOT / "data" / "reviews_final.csv", index=False)
    console.print("Saved reviews_with_sentiment.csv and reviews_final.csv")

    # ── Downstream ────────────────────────────────────────────────────────
    console.rule("BI Extraction")
    bi_outputs = run_bi(df)

    console.rule("Temporal Analysis")
    temporal_outputs = run_temporal(df)

    console.rule("Visualisations")
    save_all_static(df, bi_outputs, temporal_outputs)

    elapsed = (time.time() - start) / 60
    console.rule(f"[bold green]Done in {elapsed:.1f} min[/bold green]")
    console.print(f"""
  [green]✓[/green]  Columns in CSV:
       vader_score, vader_label              (lexicon baseline)
       distilbert_label, distilbert_score    (binary transformer baseline)
       roberta_label, roberta_score          (RoBERTa primary — 3-class)
       sentiment_score, final_sentiment      (fused)

  [cyan]→[/cyan]  Restart dashboard:  streamlit run app.py
    """)


if __name__ == "__main__":
    main()
