from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
OUTPUTS_DIR   = ROOT / "outputs"
MODELS_DIR    = OUTPUTS_DIR / "models"
PLOTS_DIR     = OUTPUTS_DIR / "plots"
REPORTS_DIR   = OUTPUTS_DIR / "reports"

RAW_CSV       = DATA_DIR / "amazon_reviews_raw.csv"
PROCESSED_CSV = DATA_DIR / "reviews_processed.csv"

for _d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics",
    "Home_and_Kitchen",
    "Clothing_Shoes_and_Jewelry",
    "Sports_and_Outdoors",
    "Grocery_and_Gourmet_Food",
    "Toys_and_Games",
]

RATING_TO_LABEL = {1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}

# ── Preprocessing ──────────────────────────────────────────────────────────────
MIN_REVIEW_LENGTH   = 30          # characters after cleaning
MAX_REVIEW_LENGTH   = 2000        # truncate very long reviews
SPACY_MODEL         = "en_core_web_sm"
BATCH_SIZE_SPACY    = 512

# ── Topic Modelling ────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # fast + accurate sentence embeddings
N_TOPICS_PER_CAT    = 10                    # BERTopic target topics per category
N_TOPICS_GLOBAL     = 20                    # cross-category topics
LDA_N_TOPICS        = 10                    # sklearn LDA topics (baseline)
LDA_MAX_ITER        = 20

# ── Sentiment ──────────────────────────────────────────────────────────────────
SENTIMENT_MODEL     = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_BATCH     = 128
VADER_COMPOUND_POS  = 0.05
VADER_COMPOUND_NEG  = -0.05

# Implicit sentiment — hedging / sarcasm / passive dissatisfaction patterns
IMPLICIT_NEG_PATTERNS = [
    r"i\s+guess\s+it",
    r"not\s+(exactly|quite|really|what\s+i\s+expected)",
    r"still\s+waiting",
    r"took\s+(forever|ages|\d+\s+weeks|\d+\s+months)",
    r"somehow\s+manages?\s+to",
    r"would\s+not\s+recommend",
    r"wish\s+i\s+(had|could|knew)",
    r"misleading\s+(description|photo|image|listing)",
    r"false\s+advertising",
    r"as\s+described\s*\?",        # rhetorical
    r"barely\s+(works?|functions?)",
    r"gave\s+it\s+a\s+chance",
    r"expected\s+(better|more)",
    r"nice\s+try",
    r"if\s+only\s+it",
]

# ── BI Extraction ──────────────────────────────────────────────────────────────
IMPACT_FREQ_WEIGHT      = 0.4    # weight of frequency in priority score
IMPACT_SENTIMENT_WEIGHT = 0.6    # weight of negative sentiment in priority score
TOP_N_COMPLAINTS        = 10
TOP_N_PRAISES           = 10

# ── Random seed ────────────────────────────────────────────────────────────────
SEED = 42
