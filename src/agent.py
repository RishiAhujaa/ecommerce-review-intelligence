"""
LLM Agent — Groq-powered Business Intelligence Narrator.

Takes structured BI outputs (priority matrix, scorecard, aspect sentiment,
complaints/praises) and generates natural-language business insights per category,
plus a cross-category executive summary.

Requires: GROQ_API_KEY environment variable (or passed directly).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd
from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CATEGORIES, REPORTS_DIR

log = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Context builders ──────────────────────────────────────────────────────────

def _sample_reviews(df: pd.DataFrame, topic: str, sentiment: str, n: int = 3) -> List[str]:
    mask = (df["bert_topic_label"] == topic) & (df["final_sentiment"] == sentiment)
    return [str(t)[:280] for t in df[mask]["text"].dropna().head(n).tolist()]


def build_category_context(df: pd.DataFrame, bi_outputs: Dict, category: str) -> dict:
    """Extract all relevant BI numbers + sample reviews for one category."""
    sub = df[df["category"] == category]

    # Basic stats
    ctx = {
        "category": category.replace("_", " "),
        "total_reviews": len(sub),
        "avg_star_rating": round(float(sub["rating"].mean()), 2),
        "pct_positive":  round(float((sub["final_sentiment"] == "Positive").mean() * 100), 1),
        "pct_neutral":   round(float((sub["final_sentiment"] == "Neutral").mean()  * 100), 1),
        "pct_negative":  round(float((sub["final_sentiment"] == "Negative").mean() * 100), 1),
        "pct_implicit_negative": round(float(sub["implicit_negative"].mean() * 100), 1),
    }

    # Priority topics
    pmat = bi_outputs.get(f"priority_{category}", pd.DataFrame())
    ctx["top_priority_topics"] = []
    if not pmat.empty:
        for _, row in pmat.head(8).iterrows():
            ctx["top_priority_topics"].append({
                "topic":          row["bert_topic_label"],
                "total_reviews":  int(row["total"]),
                "neg_ratio":      round(float(row["neg_ratio"]) * 100, 1),
                "pos_ratio":      round(float(row["pos_ratio"]) * 100, 1),
                "priority_score": round(float(row["priority_score"]), 3),
            })

    # Top complaints & praises with sample reviews
    cp_df = bi_outputs.get("complaints_praises", {}).get(category, pd.DataFrame())
    ctx["top_complaints"] = []
    ctx["top_praises"] = []
    if not cp_df.empty:
        for _, row in cp_df[cp_df["type"] == "complaint"].head(3).iterrows():
            topic = row["topic"]
            ctx["top_complaints"].append({
                "topic":          topic,
                "neg_ratio":      round(float(row.get("neg_ratio", 0)) * 100, 1),
                "count":          int(row["count"]),
                "sample_reviews": _sample_reviews(sub, topic, "Negative"),
            })
        for _, row in cp_df[cp_df["type"] == "praise"].head(3).iterrows():
            topic = row["topic"]
            ctx["top_praises"].append({
                "topic":          topic,
                "pos_ratio":      round(float(row.get("pos_ratio", 0)) * 100, 1),
                "count":          int(row["count"]),
                "sample_reviews": _sample_reviews(sub, topic, "Positive", n=2),
            })

    # Aspect sentiment
    asp_df = bi_outputs.get("aspect_sentiment", pd.DataFrame())
    ctx["aspect_sentiment"] = []
    if not asp_df.empty:
        for _, row in asp_df[asp_df["category"] == category].sort_values("neg_ratio", ascending=False).iterrows():
            ctx["aspect_sentiment"].append({
                "aspect":        row["aspect"],
                "mention_count": int(row["mention_count"]),
                "avg_sentiment": round(float(row["avg_sentiment"]), 3),
                "neg_ratio":     round(float(row["neg_ratio"]) * 100, 1),
                "pos_ratio":     round(float(row["pos_ratio"]) * 100, 1),
            })

    return ctx


# ── Prompt builders ───────────────────────────────────────────────────────────

def _format_category_prompt(ctx: dict) -> str:
    cat = ctx["category"]

    complaints_block = ""
    for c in ctx["top_complaints"]:
        complaints_block += f"\n• **{c['topic']}** — {c['neg_ratio']}% negative ({c['count']:,} reviews)\n"
        for r in c["sample_reviews"]:
            complaints_block += f'  > "{r}"\n'

    praises_block = ""
    for p in ctx["top_praises"]:
        praises_block += f"\n• **{p['topic']}** — {p['pos_ratio']}% positive ({p['count']:,} reviews)\n"
        for r in p["sample_reviews"]:
            praises_block += f'  > "{r}"\n'

    priority_block = "\n".join(
        f"  {i+1}. {t['topic']} — {t['neg_ratio']}% neg, {t['total_reviews']:,} reviews, "
        f"priority={t['priority_score']}"
        for i, t in enumerate(ctx["top_priority_topics"])
    )

    aspect_block = "\n".join(
        f"  {a['aspect']}: {a['neg_ratio']}% negative | {a['pos_ratio']}% positive "
        f"({a['mention_count']:,} mentions, avg sentiment={a['avg_sentiment']})"
        for a in ctx["aspect_sentiment"]
    )

    return f"""You are a senior e-commerce business analyst. Analyze the following data from Amazon product reviews for the **{cat}** category and produce a structured business intelligence report.

## Category: {cat}
{ctx['total_reviews']:,} reviews | Avg star rating: {ctx['avg_star_rating']}/5
Sentiment: {ctx['pct_positive']}% Positive | {ctx['pct_neutral']}% Neutral | {ctx['pct_negative']}% Negative
Implicit negative signals (dissatisfaction without explicit negative words): {ctx['pct_implicit_negative']}%

## Priority Topics (ranked by frequency × negativity — higher = fix first)
{priority_block}

## Top Complaint Topics with Customer Voice
{complaints_block}
## Top Praise Topics with Customer Voice
{praises_block}
## Aspect-Level Sentiment
{aspect_block}

---

Write a business intelligence report with exactly these four sections:

### Executive Summary
2-3 sentences covering the overall sentiment health and the single most important finding.

### Critical Issues
For each of the top 2-3 complaint topics: what customers are experiencing, why it matters commercially, likely root cause. Reference actual percentages and counts.

### Strengths
For the top 1-2 praise topics: what customers love and why it is a competitive advantage worth protecting.

### Actionable Recommendations
Exactly 3 numbered, specific, implementable recommendations. Each must state: what to do, why (data-backed), expected impact on customer satisfaction.

Rules: reference actual topic names and numbers. Write for a business stakeholder, not a data scientist. Total response under 600 words."""


def _format_executive_prompt(all_contexts: List[dict]) -> str:
    lines = []
    for ctx in all_contexts:
        top_complaint = ctx["top_complaints"][0]["topic"] if ctx["top_complaints"] else "N/A"
        top_praise    = ctx["top_praises"][0]["topic"]    if ctx["top_praises"]    else "N/A"
        lines.append(
            f"• **{ctx['category']}** ({ctx['total_reviews']:,} reviews): "
            f"{ctx['pct_positive']}% pos | {ctx['pct_negative']}% neg | "
            f"★{ctx['avg_star_rating']} | top complaint: {top_complaint} | top praise: {top_praise}"
        )

    total_reviews = sum(c["total_reviews"] for c in all_contexts)

    return f"""You are a senior e-commerce strategy consultant. Based on sentiment analysis of {total_reviews:,} Amazon reviews across {len(all_contexts)} product categories, write a cross-category executive summary.

## Category Overview
{chr(10).join(lines)}

Write a concise executive summary with exactly these sections:

### Portfolio Health
2-3 sentences: overall picture across all categories.

### Cross-Category Patterns
Issues or strengths that appear across multiple categories (e.g., delivery problems everywhere, or quality concentrated in one area). Be specific about which categories share which pattern.

### Best and Worst Performing Categories
Name the top 1-2 healthiest and the bottom 1-2 most problematic, and explain why with numbers.

### Strategic Recommendations
3 numbered recommendations that apply across categories or address the most critical systemic issues.

Keep total response under 400 words. Be specific with category names and numbers."""


# ── Gemini API call ───────────────────────────────────────────────────────────

def _groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key, http_client=httpx.Client(verify=False))


def _call_groq(prompt: str, api_key: str) -> str:
    client = _groq_client(api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _build_chat_system_context(all_contexts: List[dict]) -> str:
    """
    Build a rich system-level data dump that stays in context for the whole chat.
    The model knows all the numbers before the first user question is asked.
    """
    lines = [
        "You are a data-grounded e-commerce analyst. You have access to sentiment and topic analysis "
        "results from Amazon product reviews. Answer questions specifically using the numbers and topic "
        "names below. Do not guess or hallucinate — if data is not present, say so. Be concise.\n",
        f"Total reviews analysed: {sum(c['total_reviews'] for c in all_contexts):,}",
        f"Categories: {', '.join(c['category'] for c in all_contexts)}\n",
    ]

    for ctx in all_contexts:
        cat = ctx["category"]
        lines.append(f"── {cat} ({ctx['total_reviews']:,} reviews) ──")
        lines.append(
            f"  Sentiment: {ctx['pct_positive']}% Positive | {ctx['pct_neutral']}% Neutral | "
            f"{ctx['pct_negative']}% Negative | Avg ★{ctx['avg_star_rating']}"
        )
        lines.append(f"  Implicit negative signals: {ctx['pct_implicit_negative']}%")

        if ctx.get("top_priority_topics"):
            lines.append("  Priority topics (by frequency × negativity):")
            for t in ctx["top_priority_topics"]:
                lines.append(
                    f"    • {t['topic']}: {t['neg_ratio']}% neg, {t['pos_ratio']}% pos, "
                    f"{t['total_reviews']:,} reviews, priority={t['priority_score']}"
                )

        if ctx.get("top_complaints"):
            lines.append("  Top complaints:")
            for c in ctx["top_complaints"]:
                lines.append(f"    • {c['topic']}: {c['neg_ratio']}% negative ({c['count']:,} reviews)")
                for r in c.get("sample_reviews", [])[:2]:
                    lines.append(f'      > "{r}"')

        if ctx.get("top_praises"):
            lines.append("  Top praises:")
            for p in ctx["top_praises"]:
                lines.append(f"    • {p['topic']}: {p['pos_ratio']}% positive ({p['count']:,} reviews)")

        if ctx.get("aspect_sentiment"):
            lines.append("  Aspect sentiment:")
            for a in ctx["aspect_sentiment"]:
                lines.append(
                    f"    • {a['aspect']}: {a['neg_ratio']}% neg | {a['pos_ratio']}% pos "
                    f"({a['mention_count']:,} mentions)"
                )
        lines.append("")

    return "\n".join(lines)


def answer_question(
    all_contexts: List[dict],
    question: str,
    history: List[dict],
    api_key: str,
) -> str:
    """
    Answer a free-form question about the review data using multi-turn chat.

    history: list of {"role": "user"|"model", "content": str}
    Returns the model's reply as a string.
    """
    client = _groq_client(api_key)

    system_ctx = _build_chat_system_context(all_contexts)

    # Groq uses OpenAI-style messages with a system role
    messages = [{"role": "system", "content": system_ctx}]
    for msg in history:
        # Groq uses "assistant" not "model"
        role = "assistant" if msg["role"] == "model" else msg["role"]
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
    return response.choices[0].message.content


# ── Cache (JSON files alongside other reports) ────────────────────────────────

def _cache_path(key: str) -> Path:
    return REPORTS_DIR / f"ai_insight_{key.replace(' ', '_')}.json"


def _load_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(key: str, data: dict):
    _cache_path(key).write_text(json.dumps(data, indent=2))


# ── Public functions ──────────────────────────────────────────────────────────

def generate_category_insight(
    df: pd.DataFrame,
    bi_outputs: Dict,
    category: str,
    api_key: str,
    force: bool = False,
) -> dict:
    """
    Generate (or load cached) Gemini insight for one category.
    Returns: {category, insight_text, generated_at, context}
    """
    if not force:
        cached = _load_cache(category)
        if cached:
            log.info(f"Cache hit: {category}")
            return cached

    log.info(f"Calling Gemini for {category}...")
    ctx = build_category_context(df, bi_outputs, category)
    prompt = _format_category_prompt(ctx)
    text = _call_groq(prompt, api_key)

    result = {
        "category":      category,
        "insight_text":  text,
        "generated_at":  datetime.now().isoformat(),
        "context":       ctx,
    }
    _save_cache(category, result)
    return result


def generate_executive_summary(
    all_contexts: List[dict],
    api_key: str,
    force: bool = False,
) -> dict:
    """Generate (or load cached) cross-category executive summary."""
    key = "_executive_summary"
    if not force:
        cached = _load_cache(key)
        if cached:
            log.info("Cache hit: executive summary")
            return cached

    log.info("Calling Gemini for executive summary...")
    prompt = _format_executive_prompt(all_contexts)
    text = _call_groq(prompt, api_key)

    result = {
        "category":     key,
        "insight_text": text,
        "generated_at": datetime.now().isoformat(),
    }
    _save_cache(key, result)
    return result


def run(df: pd.DataFrame, bi_outputs: Dict, api_key: str, force: bool = False) -> Dict:
    """
    Generate insights for all categories + executive summary.
    Returns dict mapping category -> insight dict.
    """
    insights = {}
    all_contexts = []

    for cat in CATEGORIES:
        try:
            result = generate_category_insight(df, bi_outputs, cat, api_key, force)
            insights[cat] = result
            if "context" in result:
                all_contexts.append(result["context"])
        except Exception as e:
            log.error(f"{cat}: {e}")
            insights[cat] = {
                "category":     cat,
                "insight_text": f"⚠️ Error generating insight: {e}",
                "generated_at": datetime.now().isoformat(),
            }

    if all_contexts:
        try:
            insights["_executive_summary"] = generate_executive_summary(all_contexts, api_key, force)
        except Exception as e:
            log.error(f"Executive summary: {e}")

    return insights
