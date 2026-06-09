# Research Analysis: Topic Modelling & Business Intelligence from E-Commerce Reviews

## What Exists, What's Missing, and How We Go Further

---

## Paper 1 — TopicImpact: Improving Customer Feedback Analysis with Opinion Units for Topic Modeling and Star-Rating Prediction
**Authors:** Emil Häglund, Johanna Björklund
**Published:** arXiv, July 2025
**Link:** https://arxiv.org/abs/2507.13392

### What They Do
- Restructure the topic modelling pipeline to operate on **opinion units** — short, extracted phrases from raw reviews using an LLM
- The LLM assigns a sentiment score (1–10) per opinion unit before topic modelling runs
- Run BERTopic on these cleaned units instead of raw review text
- Correlate discovered topics with star ratings to predict business impact

### Limitations
- **Single domain only** — tested on one product category; no cross-category generalisation
- **LLM dependency** — extracting opinion units requires calling an LLM (expensive, API rate-limited, not offline-capable)
- **No actionable BI output** — finds topics and correlates with stars, but produces no business recommendation layer
- **Static analysis** — no temporal component; cannot detect how topics or sentiments shift over time

---

## Paper 2 — Automating Customer Feedback Analysis in E-Commerce: A Multi-Model Approach
**Authors:** Maalej, Biryuk, Wei & Panse
**Published:** Expert Systems with Applications, ScienceDirect, 2025
**Link:** https://www.sciencedirect.com/science/article/pii/S095741742504480X

### What They Do
- Build a full NLP pipeline: feedback collection → preprocessing → sentiment classification → clustering → summarisation
- Use BERT for classification and clustering for topic identification
- Use LLMs to summarise clusters into human-readable insights
- Match extracted feedback clusters to development artifacts (issue trackers, feature requests)
- Tested on Trustpilot reviews, identifies 14 key e-commerce experience aspects

### Limitations
- **No multi-category topic analysis** — treats all reviews as one blob; no category-specific topic emergence
- **No cross-domain BI** — does not compare what customers complain about across product verticals
- **Informal/noisy text handling is weak** — authors acknowledge human intervention is often needed
- **Binary or coarse sentiment** — no fine-grained aspect-level sentiment (e.g., "battery great, screen bad")
- **No real-time capability** — batch pipeline only

---

## Paper 3 — Aspect-Based Sentiment Analysis of Amazon Product Reviews Using Machine Learning Models and Hybrid Feature Engineering
**Published:** ResearchGate / Journal, 2025
**Link:** https://www.researchgate.net/publication/391916023

### What They Do
- Focus on **Aspect-Based Sentiment Analysis (ABSA)** — extract specific product aspects (price, quality, delivery) and determine sentiment per aspect
- Use hybrid feature engineering combining TF-IDF, word embeddings, and POS tagging
- Achieve up to 94.50% accuracy with Random Forest on Amazon reviews
- Compare ML models: SVM, RF, Logistic Regression, CNN

### Limitations
- **Predefined fixed aspect list** — aspects (price, quality, etc.) are manually defined; cannot discover **unknown** emerging aspects
- **Single category dataset** — trained and evaluated on one Amazon category
- **No topic modelling** — ABSA finds aspects in individual reviews but cannot surface macro-level themes across thousands of reviews
- **No BI output** — high accuracy but no business insight layer; just classification labels
- **Ignores implicit sentiment** — when customers imply dissatisfaction without explicit negative words (e.g., *"arrived in 3 weeks"*)

---

## Paper 4 — Creating Meaningful Insights from Customer Reviews: A Methodological Comparison of Topic Modeling Algorithms in Marketing Research
**Published:** Journal of Marketing Analytics, Springer Nature, 2023
**Link:** https://link.springer.com/article/10.1057/s41270-023-00256-0

### What They Do
- Systematically compare LDA, NMF, LSA, Top2Vec, and BERTopic on customer review corpora
- Evaluate which algorithm produces the most interpretable and coherent topics for marketing decisions
- Conclude BERTopic outperforms traditional methods on coherence and diversity
- Provide guidelines for practitioners choosing a topic model for marketing analytics

### Limitations
- **Comparison study only** — no novel methodology; purely evaluative
- **No sentiment integration** — topics are extracted in isolation from sentiment signal
- **No cross-category analysis** — each dataset is single-domain
- **No temporal trend analysis** — all data treated as a static snapshot
- **No downstream BI** — stops at topic labels; does not translate into pricing, inventory, or product strategy recommendations

---

## Paper 5 (Bonus) — A Novel Comprehensive Method for Customer Segmentation Based on Identifying Topics and Sentiments from Unstructured Online Product Reviews
**Published:** AIMS Big Data & Information Analytics, 2026
**Link:** https://www.aimspress.com/article/doi/10.3934/bdia.2026001

### What They Do
- Five-stage ensemble approach: TextRank preprocessing → Word2Vec topic identification → BERT+TF-IDF clause-level sentiment
- Segment customers into groups based on topic + sentiment combinations
- Validated on 22,320 reviews; achieves F1-score of 0.9433 with CatBoost
- Outputs customer segments like "quality-focused dissatisfied" or "price-sensitive satisfied"

### Limitations
- **Small, single-domain dataset** (22k reviews, one category)
- **No multi-category generalisation** — segmentation doesn't scale across product types
- **Customer segmentation only** — does not produce actionable product or business strategy recommendations
- **No temporal or trend layer** — static snapshot analysis

---

## The Gap We Are Solving

After surveying the landscape, every existing paper falls into one or more of these traps:

| Gap | Papers That Miss It |
|-----|-------------------|
| Single product category only | Papers 1, 2, 3, 4, 5 |
| No temporal / trend analysis | Papers 1, 2, 3, 4, 5 |
| No cross-category comparison of BI insights | All papers |
| No implicit sentiment detection | Papers 3, 4 |
| No actionable business recommendations output | Papers 1, 3, 4, 5 |
| LLM dependency (expensive, not reproducible) | Papers 1, 2 |

---

## Our Novel Contributions — What Makes This Project Unique

### 1. Multi-Category Comparative Topic Intelligence
**No existing paper does cross-category topic modelling.** We run topic discovery across 6 simultaneous product categories and surface: which topics are universal (appear across all categories) vs. category-specific. This is new.

### 2. Temporal Drift Detection
Using review timestamps, we track how topic prevalence and sentiment evolve over time. A topic like "battery life" may have been a major complaint in 2019 but resolved by 2023. **No paper models this drift.**

### 3. Implicit Sentiment Detection (Beyond Star Ratings)
We go beyond star ratings and explicit opinion words. Using contextual embeddings, we detect **implicit negative sentiment** — reviews where the customer doesn't say "bad" but the language signals dissatisfaction (e.g., *"still waiting after 3 weeks"*, *"I guess it works"*). This addresses a known gap in Papers 3 and 4.

### 4. Zero-LLM Topic Modelling Pipeline
Unlike TopicImpact (Paper 1), our pipeline uses **BERTopic + sentence-transformers** — powerful, locally runnable, no API calls, no cost, fully reproducible. We achieve interpretable topics without LLM dependency.

### 5. Actionable Business Intelligence Layer
Every paper stops at labels or topics. **We go further.** We produce:
- **Product complaint heatmaps** — which aspects hurt ratings the most
- **Category-level BI scorecards** — automated insight summaries per category
- **Competitive gap analysis** — what customers love in Electronics but hate in Home & Kitchen
- **Prioritisation matrix** — ranks which complaint topics have the highest business impact (high frequency × high negative sentiment = fix this first)

### 6. Unified End-to-End Dashboard
An interactive **Streamlit dashboard** that brings all of the above together for a non-technical business user — something no research paper provides as a deployable artefact.

---

## Our System Architecture (High Level)

```
204,000 Raw Reviews (6 Categories, Balanced 1★–5★)
              │
              ▼
    ┌─────────────────────┐
    │  Text Preprocessing  │  Clean, lemmatize, remove noise
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   BERTopic Modelling │  Per-category + cross-category topics
    │   + LDA (baseline)  │  Coherence scoring, topic labelling
    └─────────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  Sentiment Analysis       │  VADER (fast) + RoBERTa (accurate)
    │  + Implicit Sentiment     │  Detect implied negativity
    └──────────────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  Temporal Trend Engine    │  Topic × Sentiment over time
    └──────────────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  BI Extraction Layer      │  Prioritisation matrix, scorecards,
    │                           │  competitive gap analysis
    └──────────────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  Streamlit Dashboard      │  Interactive, business-ready
    └──────────────────────────┘
```

---

## Summary Table

| Feature | Paper 1 | Paper 2 | Paper 3 | Paper 4 | **Ours** |
|---------|---------|---------|---------|---------|---------|
| Multi-category | ✗ | ✗ | ✗ | ✗ | **✓** |
| Temporal trends | ✗ | ✗ | ✗ | ✗ | **✓** |
| Implicit sentiment | ✗ | ✗ | ✗ | ✗ | **✓** |
| No LLM dependency | ✗ | ✗ | ✓ | ✓ | **✓** |
| Actionable BI output | ✗ | Partial | ✗ | ✗ | **✓** |
| Cross-category comparison | ✗ | ✗ | ✗ | ✗ | **✓** |
| Interactive dashboard | ✗ | ✗ | ✗ | ✗ | **✓** |
| Reproducible / open source | ✗ | ✗ | ✓ | ✓ | **✓** |
