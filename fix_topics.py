"""
One-off fix: reduce BERTopic outliers (-1 topics) by reassigning each
noise review to its nearest topic using c-TF-IDF similarity.
Then re-runs BI extraction + saves updated CSVs.
No re-embedding needed — uses saved .pkl models.
"""

import pickle
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track

logging.basicConfig(level=logging.WARNING)
console = Console()

ROOT      = Path(__file__).parent
MODELS    = ROOT / "outputs" / "models"
DATA      = ROOT / "data"
REPORTS   = ROOT / "outputs" / "reports"

from config import CATEGORIES


def reduce_category_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    console.print("\n[bold cyan]── Per-Category Outlier Reduction ──[/bold cyan]")

    for cat in track(CATEGORIES, description="Reducing per-category outliers"):
        model_path = MODELS / f"bertopic_{cat}.pkl"
        if not model_path.exists():
            console.print(f"[red]Model not found: {model_path}[/red]")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        mask  = df["category"] == cat
        docs  = df.loc[mask, "lemmatised_text"].tolist()
        topics_orig = df.loc[mask, "bert_topic_id"].tolist()

        noise_before = sum(t == -1 for t in topics_orig)

        # Reassign -1 topics using c-TF-IDF similarity
        try:
            new_topics = model.reduce_outliers(
                docs, topics_orig,
                strategy="c-tf-idf",
                threshold=0.0,   # reassign ALL outliers
            )
        except Exception as e:
            console.print(f"[yellow]  {cat}: reduce_outliers failed ({e}), skipping[/yellow]")
            continue

        noise_after = sum(t == -1 for t in new_topics)

        # Update topic info with new assignments
        model.update_topics(docs, topics=new_topics)
        topic_info = model.get_topic_info()
        topic_labels = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}

        df.loc[mask, "bert_topic_id"]    = new_topics
        df.loc[mask, "bert_topic_label"] = [topic_labels.get(t, "Unknown") for t in new_topics]

        # Save updated model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        console.print(f"  [green]{cat}[/green]: noise {noise_before:,} → {noise_after:,} "
                      f"({noise_before/len(docs)*100:.1f}% → {noise_after/len(docs)*100:.1f}%)")

    return df


def reduce_global_outliers(df: pd.DataFrame) -> pd.DataFrame:
    console.print("\n[bold cyan]── Global Outlier Reduction ──[/bold cyan]")
    model_path = MODELS / "bertopic_global.pkl"
    if not model_path.exists():
        console.print("[red]Global model not found![/red]")
        return df

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    docs        = df["lemmatised_text"].tolist()
    topics_orig = df["global_topic_id"].tolist()
    noise_before = sum(t == -1 for t in topics_orig)

    try:
        new_topics = model.reduce_outliers(
            docs, topics_orig,
            strategy="c-tf-idf",
            threshold=0.0,
        )
    except Exception as e:
        console.print(f"[red]Global reduce_outliers failed: {e}[/red]")
        return df

    noise_after = sum(t == -1 for t in new_topics)

    model.update_topics(docs, topics=new_topics)
    topic_info   = model.get_topic_info()
    topic_labels = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}

    df = df.copy()
    df["global_topic_id"]    = new_topics
    df["global_topic_label"] = [topic_labels.get(t, "Unknown") for t in new_topics]

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    console.print(f"  Global noise: {noise_before:,} → {noise_after:,} "
                  f"({noise_before/len(df)*100:.1f}% → {noise_after/len(df)*100:.1f}%)")
    return df


def rerun_bi(df: pd.DataFrame):
    console.print("\n[bold cyan]── Re-running BI Extraction ──[/bold cyan]")
    from src.bi_extractor import run as run_bi
    bi_outputs = run_bi(df)
    console.print("[green]BI extraction done.[/green]")
    return bi_outputs


def rerun_temporal(df: pd.DataFrame):
    console.print("\n[bold cyan]── Re-running Temporal Analysis ──[/bold cyan]")
    from src.temporal_analysis import run as run_temporal
    temporal_outputs = run_temporal(df)
    console.print("[green]Temporal analysis done.[/green]")
    return temporal_outputs


def rerun_visualisations(df, bi_outputs, temporal_outputs):
    console.print("\n[bold cyan]── Re-generating Visualisations ──[/bold cyan]")
    from src.visualiser import save_all_static
    save_all_static(df, bi_outputs, temporal_outputs)
    console.print("[green]Visualisations saved.[/green]")


if __name__ == "__main__":
    console.rule("[bold]Topic Outlier Fix[/bold]")

    # Load the latest full dataset
    df = pd.read_csv(DATA / "reviews_final.csv")
    console.print(f"Loaded {len(df):,} reviews")

    # Step 1: reduce per-category noise
    df = reduce_category_outliers(df)

    # Step 2: reduce global noise
    df = reduce_global_outliers(df)

    # Step 3: print updated noise stats
    console.print("\n[bold]Updated noise stats:[/bold]")
    for cat in CATEGORIES:
        sub   = df[df["category"] == cat]
        noise = (sub["bert_topic_id"] == -1).mean() * 100
        console.print(f"  {cat}: {noise:.1f}% noise remaining")
    global_noise = (df["global_topic_id"] == -1).mean() * 100
    console.print(f"  Global: {global_noise:.1f}% noise remaining")

    # Step 4: save updated CSVs
    df.to_csv(DATA / "reviews_final.csv", index=False)
    df.to_csv(DATA / "reviews_with_topics.csv", index=False)
    console.print("\n[green]Saved updated reviews_final.csv and reviews_with_topics.csv[/green]")

    # Step 5: re-run downstream steps
    bi_outputs       = rerun_bi(df)
    temporal_outputs = rerun_temporal(df)
    rerun_visualisations(df, bi_outputs, temporal_outputs)

    console.rule("[bold green]Done[/bold green]")
