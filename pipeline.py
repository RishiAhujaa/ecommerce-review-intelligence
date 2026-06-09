"""
Main pipeline orchestrator.
Run this to execute the full end-to-end analysis:
  1. Preprocessing
  2. Topic Modelling (BERTopic per-category + global, LDA baseline)
  3. Sentiment Analysis (VADER + RoBERTa + implicit fusion)
  4. Business Intelligence Extraction
  5. Temporal Analysis
  6. Save all visualisations
"""

import logging
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
console = Console()

ROOT = Path(__file__).parent

from config import RAW_CSV, PROCESSED_CSV, REPORTS_DIR, PLOTS_DIR
from src.preprocessor    import Preprocessor
from src.topic_model     import TopicEngine
from src.sentiment       import run_vader, RoBERTaSentiment, fuse_sentiment
from src.bi_extractor    import run as run_bi
from src.temporal_analysis import run as run_temporal
from src.visualiser      import save_all_static


def _banner(msg: str):
    console.rule(f"[bold cyan]{msg}[/bold cyan]")


def run_pipeline(skip_preprocessing: bool = False, skip_topics: bool = False):
    total_start = time.time()

    # ── STEP 1: Preprocessing ────────────────────────────────────────────────────
    _banner("STEP 1 — Text Preprocessing")
    processed_path = ROOT / "data" / "reviews_processed.csv"

    if skip_preprocessing and processed_path.exists():
        console.print("[yellow]Skipping preprocessing — loading cached file.[/yellow]")
        df = pd.read_csv(processed_path)
    else:
        df_raw = pd.read_csv(RAW_CSV)
        console.print(f"Loaded raw dataset: [bold]{df_raw.shape}[/bold]")
        prep = Preprocessor()
        df = prep.process(df_raw)
        df.to_csv(processed_path, index=False)
        console.print(f"Processed data saved -> {processed_path}")

    console.print(f"[green]Step 1 done. Shape: {df.shape}[/green]")

    # ── STEP 2: Topic Modelling ──────────────────────────────────────────────────
    _banner("STEP 2 — Topic Modelling")
    topics_path = ROOT / "data" / "reviews_with_topics.csv"

    if skip_topics and topics_path.exists():
        console.print("[yellow]Skipping topic modelling — loading cached file.[/yellow]")
        df = pd.read_csv(topics_path)
    else:
        engine = TopicEngine()
        df = engine.fit_per_category(df)
        df = engine.fit_global(df)
        engine.fit_lda_per_category(df)
        df.to_csv(topics_path, index=False)
        console.print(f"Topic data saved -> {topics_path}")

    console.print(f"[green]Step 2 done. Topics discovered globally: {df['global_topic_id'].nunique()}[/green]")

    # ── STEP 3: Sentiment Analysis ───────────────────────────────────────────────
    _banner("STEP 3 — Sentiment Analysis")
    sentiment_path = ROOT / "data" / "reviews_with_sentiment.csv"

    vader_df  = run_vader(df["clean_text"].tolist())
    df        = pd.concat([df.reset_index(drop=True), vader_df], axis=1)

    roberta   = RoBERTaSentiment()
    roberta_df = roberta.predict(df["clean_text"].tolist())
    df        = pd.concat([df.reset_index(drop=True), roberta_df], axis=1)

    df        = fuse_sentiment(df)
    df.to_csv(sentiment_path, index=False)
    console.print(f"Sentiment data saved -> {sentiment_path}")
    console.print(f"[green]Step 3 done. Distribution:[/green]")
    console.print(df["final_sentiment"].value_counts().to_string())

    # Save final enriched dataset
    final_path = ROOT / "data" / "reviews_final.csv"
    df.to_csv(final_path, index=False)

    # ── STEP 4: BI Extraction ────────────────────────────────────────────────────
    _banner("STEP 4 — Business Intelligence Extraction")
    bi_outputs = run_bi(df)
    console.print(f"[green]Step 4 done. Reports saved to {REPORTS_DIR}[/green]")

    # Print scorecard
    if "scorecard" in bi_outputs:
        sc = bi_outputs["scorecard"]
        table = Table(title="Category Scorecard", show_header=True, header_style="bold magenta")
        for col in sc.columns:
            table.add_column(col, justify="right")
        for _, row in sc.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)

    # ── STEP 5: Temporal Analysis ─────────────────────────────────────────────────
    _banner("STEP 5 — Temporal Analysis")
    temporal_outputs = run_temporal(df)
    console.print(f"[green]Step 5 done.[/green]")

    # ── STEP 6: Visualisations ────────────────────────────────────────────────────
    _banner("STEP 6 — Generating Visualisations")
    save_all_static(df, bi_outputs, temporal_outputs)
    console.print(f"[green]Step 6 done. Plots saved to {PLOTS_DIR}[/green]")

    # ── Summary ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    _banner(f"Pipeline Complete in {elapsed/60:.1f} minutes")
    console.print(f"""
    [bold green]✓[/bold green]  Processed reviews  : {len(df):,}
    [bold green]✓[/bold green]  Categories         : {df['category'].nunique()}
    [bold green]✓[/bold green]  Global topics      : {df['global_topic_id'].nunique()}
    [bold green]✓[/bold green]  Reports            : {REPORTS_DIR}
    [bold green]✓[/bold green]  Plots              : {PLOTS_DIR}
    [bold cyan]→[/bold cyan]  Launch dashboard   : [bold]streamlit run app.py[/bold]
    """)

    return df, bi_outputs, temporal_outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full Topic Modelling & BI pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing if cached file exists")
    parser.add_argument("--skip-topics",        action="store_true", help="Skip topic modelling if cached file exists")
    args = parser.parse_args()

    run_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        skip_topics=args.skip_topics,
    )
