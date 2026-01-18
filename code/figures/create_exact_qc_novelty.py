#!/usr/bin/env python3
"""
Calculate EXACT QC + Novelty numbers by matching sequence IDs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path('/Users/mcclainthiel/Downloads/results_Exp14_Aggressive')
PUB_DIR = BASE_DIR / 'publication'
BLAST_DIR = BASE_DIR / 'ncbi_blast'

MODEL_ORDER = ['Base', 'SFT', 'RL', 'SFT+RL']
MODEL_COLORS = {'Base': '#2E4057', 'SFT': '#8B5CF6', 'RL': '#E11D48', 'SFT+RL': '#F97316'}

MODEL_DIRS = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL', 'SFT+RL': 'SFT_GRPO'}
BLAST_FILES = {
    'Base': 'Base_blast_results.tsv',
    'SFT': 'SFT_blast_results.tsv',
    'RL': 'RL_blast_results.tsv',
    'SFT+RL': 'SFT_GRPO_blast_results.tsv'
}
BLAST_PREFIXES = {'Base': 'Base_', 'SFT': 'SFT_', 'RL': 'RL_', 'SFT+RL': 'SFT_GRPO_'}

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
sns.set_theme(style="white")

def classify_sequence(identity, coverage):
    """Classify based on NCBI BLAST results."""
    if identity >= 99 and coverage >= 95:
        return 'Exists'
    elif identity >= 95 and coverage >= 80:
        return 'Similar'
    else:
        return 'Novel'

def get_passed_ids(model):
    """Get set of sequence IDs that passed QC."""
    model_dir = MODEL_DIRS[model]
    passed_path = BASE_DIR / 'qc' / model_dir / 'passed.csv'
    if not passed_path.exists():
        return set()
    df = pd.read_csv(passed_path)
    return set(df['Plasmid_ID'].values)

def get_blast_classifications(model):
    """Get dict of sequence_id -> classification from BLAST results."""
    filepath = BLAST_DIR / BLAST_FILES[model]
    prefix = BLAST_PREFIXES[model]

    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath, sep='\t')

    # Get best hit per query (highest combined score)
    df['score'] = df['pct_identity'] * df['query_coverage']
    best_hits = df.loc[df.groupby('query_id')['score'].idxmax()]

    classifications = {}
    for _, row in best_hits.iterrows():
        # Remove prefix to get base seq_id
        seq_id = row['query_id'].replace(prefix, '')
        cat = classify_sequence(row['pct_identity'], row['query_coverage'])
        classifications[seq_id] = cat

    return classifications

def main():
    results = []
    total_per_model = 100

    print("=" * 60)
    print("EXACT QC + Novelty Calculation")
    print("=" * 60)

    for model in MODEL_ORDER:
        passed_ids = get_passed_ids(model)
        blast_classifications = get_blast_classifications(model)

        # Count exact overlaps
        passed_novel = 0
        passed_similar = 0
        passed_exists = 0
        passed_not_blasted = 0

        for seq_id in passed_ids:
            if seq_id in blast_classifications:
                cat = blast_classifications[seq_id]
                if cat == 'Novel':
                    passed_novel += 1
                elif cat == 'Similar':
                    passed_similar += 1
                else:
                    passed_exists += 1
            else:
                passed_not_blasted += 1

        n_passed = len(passed_ids)
        n_failed = total_per_model - n_passed

        print(f"\n{model}:")
        print(f"  Total: {total_per_model}")
        print(f"  Passed QC: {n_passed}")
        print(f"    - Novel: {passed_novel}")
        print(f"    - Similar: {passed_similar}")
        print(f"    - Exists: {passed_exists}")
        print(f"    - Not BLASTed: {passed_not_blasted}")
        print(f"  Failed QC: {n_failed}")

        results.append({
            'Model': model,
            'Total': total_per_model,
            'Passed QC': n_passed,
            'Failed QC': n_failed,
            'Passed & Novel': passed_novel,
            'Passed & Similar': passed_similar,
            'Passed & Exists': passed_exists,
            'Passed (not BLASTed)': passed_not_blasted
        })

    df = pd.DataFrame(results)
    df = df.set_index('Model').reindex(MODEL_ORDER)

    print("\n" + "=" * 60)
    print("Summary Table:")
    print(df)

    # Save data
    df.to_csv(PUB_DIR / 'qc_novelty_exact.csv')

    # ===== FIGURE: Stacked bar =====
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(MODEL_ORDER))
    width = 0.6

    # For sequences that weren't BLASTed, we'll show them as "Passed (unknown)"
    # Stack order from bottom: Failed, Exists, Similar, Not BLASTed, Novel

    categories = [
        ('Failed QC', '#EF4444'),
        ('Passed & Exists', '#DC2626'),
        ('Passed & Similar', '#F59E0B'),
        ('Passed (not BLASTed)', '#9CA3AF'),
        ('Passed & Novel', '#22C55E')
    ]

    bottom = np.zeros(len(MODEL_ORDER))

    for cat_name, color in categories:
        values = df[cat_name].values
        bars = ax.bar(x, values, width, bottom=bottom, label=cat_name,
                      color=color, edgecolor='white', linewidth=1)

        # Add count labels for values > 3
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 3:
                text_color = 'white' if cat_name != 'Passed & Novel' else 'black'
                ax.text(x[i], bot + val/2, str(int(val)),
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color=text_color)

        bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Sequences (out of 100)', fontsize=12)
    ax.set_title('QC Pass Rate and NCBI Novelty Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig21_qc_novelty_exact.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig21_qc_novelty_exact.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: fig21_qc_novelty_exact")

if __name__ == '__main__':
    main()
