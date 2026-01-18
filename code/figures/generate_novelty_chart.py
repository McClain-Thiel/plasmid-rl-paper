#!/usr/bin/env python3
"""
Generate novelty classification chart based on NCBI BLAST results.
Categories:
- Exists: identity >= 99% AND coverage >= 95%
- Similar: identity >= 95% AND coverage >= 80%
- Novel: anything else
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
BASE_DIR = Path('/Users/mcclainthiel/Downloads/results_Exp14_Aggressive')
PUB_DIR = BASE_DIR / 'publication'
BLAST_DIR = BASE_DIR / 'ncbi_blast'

MODEL_ORDER = ['Base', 'SFT', 'RL', 'SFT+RL']
MODEL_FILES = {
    'Base': 'Base_blast_results.tsv',
    'SFT': 'SFT_blast_results.tsv',
    'RL': 'RL_blast_results.tsv',
    'SFT+RL': 'SFT_GRPO_blast_results.tsv'
}

MODEL_COLORS = {'Base': '#2E4057', 'SFT': '#8B5CF6', 'RL': '#E11D48', 'SFT+RL': '#F97316'}

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
sns.set_theme(style="white")

def classify_sequence(identity, coverage):
    """Classify a sequence based on best BLAST hit."""
    if identity >= 99 and coverage >= 95:
        return 'Exists'
    elif identity >= 95 and coverage >= 80:
        return 'Similar'
    else:
        return 'Novel'

def process_blast_results(filepath):
    """Process BLAST results and get best hit per query."""
    df = pd.read_csv(filepath, sep='\t')

    # Get best hit per query (highest identity * coverage product)
    df['score'] = df['pct_identity'] * df['query_coverage']
    best_hits = df.loc[df.groupby('query_id')['score'].idxmax()]

    # Classify each sequence
    classifications = []
    for _, row in best_hits.iterrows():
        cat = classify_sequence(row['pct_identity'], row['query_coverage'])
        classifications.append({
            'query_id': row['query_id'],
            'pct_identity': row['pct_identity'],
            'query_coverage': row['query_coverage'],
            'category': cat
        })

    return pd.DataFrame(classifications)

def main():
    print("Processing BLAST results...")

    all_results = []

    for model, filename in MODEL_FILES.items():
        filepath = BLAST_DIR / filename
        if filepath.exists():
            results = process_blast_results(filepath)
            results['model'] = model
            all_results.append(results)
            print(f"  {model}: {len(results)} sequences")
        else:
            print(f"  {model}: File not found!")

    df = pd.concat(all_results, ignore_index=True)

    # Count by model and category
    counts = df.groupby(['model', 'category']).size().unstack(fill_value=0)

    # Ensure all categories exist
    for cat in ['Exists', 'Similar', 'Novel']:
        if cat not in counts.columns:
            counts[cat] = 0

    counts = counts[['Exists', 'Similar', 'Novel']]  # Order columns
    counts = counts.reindex(MODEL_ORDER)  # Order rows

    print("\nCounts by model and category:")
    print(counts)

    # Calculate percentages
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    print("\nPercentages:")
    print(pcts.round(1))

    # Save data
    counts.to_csv(PUB_DIR / 'novelty_counts.csv')
    pcts.to_csv(PUB_DIR / 'novelty_percentages.csv')

    # ===== FIGURE 1: Stacked bar chart (counts) =====
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(MODEL_ORDER))
    width = 0.6

    colors = {'Exists': '#EF4444', 'Similar': '#F59E0B', 'Novel': '#22C55E'}

    bottom = np.zeros(len(MODEL_ORDER))
    for cat in ['Exists', 'Similar', 'Novel']:
        values = counts[cat].values
        bars = ax.bar(x, values, width, bottom=bottom, label=cat,
                      color=colors[cat], edgecolor='white', linewidth=1)

        # Add count labels on bars
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(x[i], bot + val/2, str(int(val)),
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if cat != 'Novel' else 'black')

        bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Sequences', fontsize=12)
    ax.set_title('Sequence Novelty Classification (NCBI BLAST)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig18_novelty_stacked.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig18_novelty_stacked.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: fig18_novelty_stacked")

    # ===== FIGURE 2: Stacked bar chart (percentages) =====
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(MODEL_ORDER))
    for cat in ['Exists', 'Similar', 'Novel']:
        values = pcts[cat].values
        bars = ax.bar(x, values, width, bottom=bottom, label=cat,
                      color=colors[cat], edgecolor='white', linewidth=1)

        # Add percentage labels
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:  # Only label if > 5%
                ax.text(x[i], bot + val/2, f'{val:.0f}%',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if cat != 'Novel' else 'black')

        bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage of Sequences', fontsize=12)
    ax.set_title('Sequence Novelty Classification (NCBI BLAST)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig19_novelty_percentage.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig19_novelty_percentage.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig19_novelty_percentage")

    # ===== FIGURE 3: Grouped bar chart =====
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    for i, cat in enumerate(['Exists', 'Similar', 'Novel']):
        offset = (i - 1) * width
        values = pcts[cat].values
        bars = ax.bar(x + offset, values, width, label=cat, color=colors[cat],
                      edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage of Sequences', fontsize=12)
    ax.set_title('Sequence Novelty by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig20_novelty_grouped.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig20_novelty_grouped.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig20_novelty_grouped")

    print("\n" + "="*50)
    print("Summary Table:")
    print("="*50)
    for model in MODEL_ORDER:
        row = pcts.loc[model]
        print(f"{model:10s}: Exists={row['Exists']:5.1f}%, Similar={row['Similar']:5.1f}%, Novel={row['Novel']:5.1f}%")

if __name__ == '__main__':
    main()
