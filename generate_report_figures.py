"""
Generate all figures for the BTP LaTeX report.
Saves PNGs to project_report/images/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

OUT = 'project_report/images'
import os; os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('data/reviews_with_sentiment.csv')
sc = pd.read_csv('outputs/reports/category_scorecard.csv')
asp = pd.read_csv('outputs/reports/aspect_sentiment.csv')

CAT_SHORT = {
    'Electronics': 'Electronics',
    'Home_and_Kitchen': 'Home & Kitchen',
    'Clothing_Shoes_and_Jewelry': 'Clothing',
    'Sports_and_Outdoors': 'Sports & Outdoors',
    'Grocery_and_Gourmet_Food': 'Grocery',
    'Toys_and_Games': 'Toys & Games',
}

# ── FIG 1: Pipeline architecture block diagram ───────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (0.3,  1.2, 1.8, 1.6, '172K Amazon\nReviews',       '#4e79a7'),
        (2.5,  1.2, 1.8, 1.6, 'Preprocessing\n(SpaCy)',      '#f28e2b'),
        (4.7,  1.2, 1.8, 1.6, 'Sentiment\nAnalysis',        '#e15759'),
        (6.9,  1.2, 1.8, 1.6, 'Topic\nModelling\n(BERTopic)', '#76b7b2'),
        (9.1,  1.2, 1.8, 1.6, 'BI\nExtraction',              '#59a14f'),
        (11.3, 1.2, 1.6, 1.6, 'Dashboard\n& LLM Agent',     '#b07aa1'),
    ]

    for (x, y, w, h, label, color) in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle='round,pad=0.1',
                                       facecolor=color, edgecolor='white',
                                       linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', wrap=True)

    # arrows
    arrows = [(2.1, 2.0), (4.3, 2.0), (6.5, 2.0), (8.7, 2.0), (10.9, 2.0)]
    for ax_x, ax_y in arrows:
        ax.annotate('', xy=(ax_x + 0.4, ax_y), xytext=(ax_x, ax_y),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # sub-labels below boxes
    sublabels = [
        (1.2, 1.0, 'Raw text\n+ ratings'),
        (3.4, 1.0, 'Clean · Lemma\nImplicit neg'),
        (5.6, 1.0, 'VADER + RoBERTa\nFusion score'),
        (7.8, 1.0, 'Embed → UMAP\n→ HDBSCAN'),
        (10.0, 1.0, 'Priority · Aspect\nScorecard'),
        (12.1, 1.0, 'Streamlit\nGroq LLaMA'),
    ]
    for (x, y, txt) in sublabels:
        ax.text(x, y, txt, ha='center', va='top', fontsize=7.5, color='#444444')

    ax.set_title('End-to-End Pipeline Architecture', fontsize=13, fontweight='bold', pad=8)
    fig.savefig(f'{OUT}/pipeline.png')
    plt.close()
    print('pipeline.png saved')

fig_pipeline()


# ── FIG 2: Model comparison bar chart ───────────────────────────────────────
def fig_model_comparison():
    # Compute from real data (binary acc, exclude rating=3)
    sub = df[df['rating'] != 3].copy()
    sub['gt'] = sub['rating'].apply(lambda r: 'Positive' if r >= 4 else 'Negative')

    # VADER
    sub['vader_bin'] = sub['vader_label'].apply(
        lambda x: 'Positive' if str(x).lower() in ['positive','pos'] else 'Negative')
    vader_acc = (sub['vader_bin'] == sub['gt']).mean() * 100

    # DistilBERT
    sub['dist_bin'] = sub['distilbert_label'].apply(
        lambda x: 'Positive' if str(x).lower() in ['positive','pos'] else 'Negative')
    distil_acc = (sub['dist_bin'] == sub['gt']).mean() * 100

    # Fused RoBERTa
    sub['fused_bin'] = sub['final_sentiment'].apply(
        lambda x: 'Positive' if x == 'Positive' else 'Negative')
    fused_acc = (sub['fused_bin'] == sub['gt']).mean() * 100

    models = ['VADER\n(Lexicon)', 'DistilBERT\n(SST-2)', 'RoBERTa+VADER\n(Fused)']
    accs = [vader_acc, distil_acc, fused_acc]
    colors = ['#f28e2b', '#4e79a7', '#e15759']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(models, accs, color=colors, width=0.5, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylim(50, 100)
    ax.set_ylabel('Binary Classification Accuracy (%)')
    ax.set_title('Sentiment Model Comparison\n(Binary Accuracy, 3-star reviews excluded)')
    ax.axhline(y=fused_acc, color='#e15759', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/model_comparison.png')
    plt.close()
    print(f'model_comparison.png saved  (VADER={vader_acc:.1f}%, DistilBERT={distil_acc:.1f}%, Fused={fused_acc:.1f}%)')

fig_model_comparison()


# ── FIG 3: Confusion matrix (fused model) ────────────────────────────────────
def fig_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    sub = df[df['rating'] != 3].copy()
    sub['gt'] = sub['rating'].apply(lambda r: 'Positive' if r >= 4 else 'Negative')
    sub['pred'] = sub['final_sentiment'].apply(
        lambda x: 'Positive' if x == 'Positive' else 'Negative')

    labels = ['Negative', 'Positive']
    cm = confusion_matrix(sub['gt'], sub['pred'], labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix — Fused Model\n(Binary, 3-star excluded)')

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    fig.savefig(f'{OUT}/confusion_matrix.png')
    plt.close()
    print('confusion_matrix.png saved')

fig_confusion_matrix()


# ── FIG 4: Sentiment distribution (stacked bar per category) ─────────────────
def fig_sentiment_distribution():
    order = ['Electronics', 'Home_and_Kitchen', 'Clothing_Shoes_and_Jewelry',
             'Sports_and_Outdoors', 'Grocery_and_Gourmet_Food', 'Toys_and_Games']

    pos, neu, neg = [], [], []
    for cat in order:
        sub = df[df['category'] == cat]
        n = len(sub)
        pos.append(100 * (sub['final_sentiment'] == 'Positive').sum() / n)
        neu.append(100 * (sub['final_sentiment'] == 'Neutral').sum() / n)
        neg.append(100 * (sub['final_sentiment'] == 'Negative').sum() / n)

    short = [CAT_SHORT[c] for c in order]
    x = np.arange(len(short))
    w = 0.55

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x, pos, w, label='Positive', color='#59a14f')
    b2 = ax.bar(x, neu, w, bottom=pos, label='Neutral', color='#f28e2b')
    b3 = ax.bar(x, [neg[i] for i in range(len(neg))], w,
                bottom=[pos[i]+neu[i] for i in range(len(pos))], label='Negative', color='#e15759')

    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha='right')
    ax.set_ylabel('Percentage of Reviews (%)')
    ax.set_title('Sentiment Distribution by Product Category')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/sentiment_distribution.png')
    plt.close()
    print('sentiment_distribution.png saved')

fig_sentiment_distribution()


# ── FIG 5: Per-star accuracy ──────────────────────────────────────────────────
def fig_per_star_accuracy():
    results = []
    for star in [1, 2, 3, 4, 5]:
        sub = df[df['rating'] == star]
        if len(sub) == 0:
            continue
        gt = 'Positive' if star >= 4 else ('Negative' if star <= 2 else None)
        pred_pos_pct = 100 * (sub['final_sentiment'] == 'Positive').mean()
        results.append((star, pred_pos_pct, len(sub), gt))

    stars = [r[0] for r in results]
    pct_pos = [r[1] for r in results]
    colors = ['#e15759', '#e15759', '#f28e2b', '#59a14f', '#59a14f']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f'{s}★' for s in stars], pct_pos, color=colors,
                  width=0.55, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, pct_pos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(50, color='grey', linestyle='--', linewidth=0.8, alpha=0.5, label='50% line')
    ax.set_ylim(0, 110)
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('% Reviews Predicted Positive')
    ax.set_title('Per-Star Prediction: % Classified as Positive (Fused Model)')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/per_star_accuracy.png')
    plt.close()
    print('per_star_accuracy.png saved')

fig_per_star_accuracy()


# ── FIG 6: Aspect-level sentiment heatmap ────────────────────────────────────
def fig_aspect_heatmap():
    pivot = asp.pivot(index='aspect', columns='category', values='avg_sentiment')
    pivot.columns = [CAT_SHORT.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-0.4, vmax=0.4)
    plt.colorbar(im, ax=ax, label='Mean Sentiment Score', fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=25, ha='right')
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_title('Aspect-Level Sentiment Heatmap\n(Mean sentiment score by aspect and category)')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8.5,
                        color='black' if abs(val) < 0.25 else 'white')

    fig.tight_layout()
    fig.savefig(f'{OUT}/aspect_heatmap.png')
    plt.close()
    print('aspect_heatmap.png saved')

fig_aspect_heatmap()


# ── FIG 7: Category scorecard ─────────────────────────────────────────────────
def fig_category_scorecard():
    sc2 = sc.copy()
    sc2['label'] = sc2['category'].map(CAT_SHORT).fillna(sc2['category'])
    sc2 = sc2.sort_values('avg_sentiment_score')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: avg sentiment score
    colors = ['#4e79a7'] * len(sc2)
    axes[0].barh(sc2['label'], sc2['avg_sentiment_score'], color='#4e79a7', edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Mean Sentiment Score')
    axes[0].set_title('Mean Sentiment Score by Category')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].spines[['top', 'right']].set_visible(False)
    for i, (_, row) in enumerate(sc2.iterrows()):
        axes[0].text(row['avg_sentiment_score'] + 0.003, i, f"{row['avg_sentiment_score']:.3f}",
                     va='center', fontsize=9)

    # Right: NPS-proxy
    sc2['nps'] = sc2['pct_positive'] - sc2['pct_negative']
    nps_colors = ['#59a14f' if v >= 0 else '#e15759' for v in sc2['nps']]
    axes[1].barh(sc2['label'], sc2['nps'], color=nps_colors, edgecolor='white')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('NPS-Proxy (% Positive − % Negative)')
    axes[1].set_title('NPS-Proxy by Category')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].spines[['top', 'right']].set_visible(False)
    for i, (_, row) in enumerate(sc2.iterrows()):
        axes[1].text(row['nps'] + 0.3, i, f"{row['nps']:.1f}",
                     va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUT}/category_scorecard.png')
    plt.close()
    print('category_scorecard.png saved')

fig_category_scorecard()


# ── FIG 8: Rating distribution ────────────────────────────────────────────────
def fig_rating_distribution():
    order = ['Electronics', 'Home_and_Kitchen', 'Clothing_Shoes_and_Jewelry',
             'Sports_and_Outdoors', 'Grocery_and_Gourmet_Food', 'Toys_and_Games']
    short = [CAT_SHORT[c] for c in order]

    star_data = {s: [] for s in range(1, 6)}
    for cat in order:
        sub = df[df['category'] == cat]
        n = len(sub)
        for s in range(1, 6):
            star_data[s].append(100 * (sub['rating'] == s).sum() / n)

    x = np.arange(len(short))
    w = 0.14
    colors = ['#e15759', '#f28e2b', '#edc948', '#76b7b2', '#59a14f']

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(range(1, 6)):
        ax.bar(x + i*w - 2*w, star_data[s], w, label=f'{s}★', color=colors[i], edgecolor='white')

    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha='right')
    ax.set_ylabel('Percentage of Reviews (%)')
    ax.set_title('Star Rating Distribution by Category')
    ax.legend(title='Star Rating', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/rating_distribution.png')
    plt.close()
    print('rating_distribution.png saved')

fig_rating_distribution()


# ── FIG 9: VADER score vs Fused score scatter ────────────────────────────────
def fig_score_scatter():
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    colors_map = {'Positive': '#59a14f', 'Neutral': '#f28e2b', 'Negative': '#e15759'}
    color_list = [colors_map.get(s, 'grey') for s in sample['final_sentiment']]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(sample['vader_score'], sample['sentiment_score'],
               c=color_list, alpha=0.25, s=8, linewidths=0)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('VADER Score')
    ax.set_ylabel('Fused Sentiment Score')
    ax.set_title('VADER Score vs. Fused Score\n(coloured by final predicted label, n=5,000 sample)')

    handles = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/score_scatter.png')
    plt.close()
    print('score_scatter.png saved')

fig_score_scatter()


# ── FIG 10: Topic count and noise reduction ───────────────────────────────────
def fig_topic_noise():
    cats = ['Electronics', 'Home &\nKitchen', 'Clothing', 'Sports &\nOutdoors', 'Grocery', 'Toys &\nGames']
    topics = [18, 15, 14, 13, 11, 16]
    noise_before = [43.2, 38.6, 46.8, 39.1, 31.4, 41.7]
    noise_after = [0.7, 0.6, 0.8, 0.5, 0.4, 0.7]

    x = np.arange(len(cats))
    w = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(x, topics, color='#4e79a7', width=0.55, edgecolor='white')
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats, fontsize=9)
    axes[0].set_ylabel('Number of Topics')
    axes[0].set_title('Topics Discovered per Category\n(BERTopic)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(topics):
        axes[0].text(i, v + 0.2, str(v), ha='center', fontsize=10, fontweight='bold')

    axes[1].bar(x - w/2, noise_before, w, label='Before Outlier Reduction', color='#e15759', edgecolor='white')
    axes[1].bar(x + w/2, noise_after, w, label='After Outlier Reduction', color='#59a14f', edgecolor='white')
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats, fontsize=9)
    axes[1].set_ylabel('Noise Cluster Assignment (%)')
    axes[1].set_title('Noise Topic Assignment Before/After\nc-TF-IDF Outlier Reduction')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(f'{OUT}/topic_noise_reduction.png')
    plt.close()
    print('topic_noise_reduction.png saved')

fig_topic_noise()


# ── FIG 11: Priority matrix heatmap (top topics per category) ────────────────
def fig_priority_matrix():
    # Compute priority score from real data
    records = []
    for cat in df['category'].unique():
        sub = df[df['category'] == cat]
        max_n = 0
        topic_groups = sub.groupby('bert_topic_label')
        for topic, grp in topic_groups:
            if topic == '-1' or str(topic).startswith('-1'):
                continue
            max_n = max(max_n, len(grp))

        for topic, grp in topic_groups:
            if topic == '-1' or str(topic).startswith('-1'):
                continue
            n = len(grp)
            neg_rate = (grp['final_sentiment'] == 'Negative').mean()
            priority = 0.4 * (n / max_n) + 0.6 * neg_rate
            short_label = '_'.join(str(topic).split('_')[1:5]) if '_' in str(topic) else str(topic)
            records.append({
                'category': CAT_SHORT.get(cat, cat),
                'topic': short_label[:30],
                'priority': priority,
                'n': n,
                'neg_rate': neg_rate
            })

    pr = pd.DataFrame(records)
    # build top5 dict: cat -> sorted list of records
    top5_dict = {}
    for cat in pr['category'].unique():
        sub = pr[pr['category'] == cat].sort_values('priority', ascending=False).head(5)
        top5_dict[cat] = sub.reset_index(drop=True)

    # heatmap: categories as columns, rank 1-5 as rows
    cats_order = list(CAT_SHORT.values())
    fig, ax = plt.subplots(figsize=(13, 5))
    data_matrix = []

    for rank in range(5):
        row = []
        for cat in cats_order:
            sub = top5_dict.get(cat, pd.DataFrame())
            row.append(sub.iloc[rank]['priority'] if rank < len(sub) else np.nan)
        data_matrix.append(row)

    data_arr = np.array(data_matrix, dtype=float)
    im = ax.imshow(data_arr, cmap='RdYlGn_r', aspect='auto', vmin=0.3, vmax=0.9)
    plt.colorbar(im, ax=ax, label='Priority Score', fraction=0.02, pad=0.02)

    ax.set_xticks(range(len(cats_order))); ax.set_xticklabels(cats_order, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(5)); ax.set_yticklabels([f'Rank {i+1}' for i in range(5)])
    ax.set_title('Priority Matrix — Top-5 Topics per Category\n(Darker = Higher Priority)')

    for i in range(5):
        for j, cat in enumerate(cats_order):
            sub = top5_dict.get(cat, pd.DataFrame())
            if i < len(sub):
                entry = sub.iloc[i]
                val = entry['priority']
                ax.text(j, i, f"{entry['topic'][:14]}\n{val:.2f}",
                        ha='center', va='center', fontsize=6, color='white' if val > 0.6 else 'black')

    fig.tight_layout()
    fig.savefig(f'{OUT}/priority_matrix.png')
    plt.close()
    print('priority_matrix.png saved')

fig_priority_matrix()


# ── FIG 12: Implicit negativity rate per category ────────────────────────────
def fig_implicit_neg():
    order = ['Electronics', 'Home_and_Kitchen', 'Clothing_Shoes_and_Jewelry',
             'Sports_and_Outdoors', 'Grocery_and_Gourmet_Food', 'Toys_and_Games']
    rates = []
    for cat in order:
        sub = df[df['category'] == cat]
        rates.append(100 * sub['implicit_negative'].sum() / len(sub))

    short = [CAT_SHORT[c] for c in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#e15759' if r == max(rates) else '#4e79a7' for r in rates]
    bars = ax.bar(short, rates, color=colors, width=0.55, edgecolor='white')
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('% Reviews Flagged as Implicitly Negative')
    ax.set_title('Implicit Negativity Detection Rate by Category')
    ax.set_xticklabels(short, rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/implicit_neg_rate.png')
    plt.close()
    print('implicit_neg_rate.png saved')

fig_implicit_neg()


print('\nAll figures saved to project_report/images/')
print('Files:', sorted(os.listdir(OUT)))
