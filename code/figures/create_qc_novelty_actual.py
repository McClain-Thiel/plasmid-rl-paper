#!/usr/bin/env python3
"""
Create chart showing sequences that both passed QC AND are novel.
Uses ACTUAL numbers by matching sequence IDs between QC and BLAST results.
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
BLAST_FILES = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL', 'SFT+RL': 'SFT_GRPO'}

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
sns.set_theme(style="white")

def classify_sequence(identity, coverage):
    """Classify based on NCBI BLAST thresholds."""
    if identity >= 99 and coverage >= 95:
        return 'Exists'
    elif identity >= 95 and coverage >= 80:
        return 'Similar'
    else:
        return 'Novel'

def get_blast_classifications(model):
    """Get novelty classification for each BLASTed sequence."""
    blast_prefix = BLAST_FILES[model]
    filepath = BLAST_DIR / f'{blast_prefix}_blast_results.tsv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath, sep='\t')
    df['score'] = df['pct_identity'] * df['query_coverage']
    best_hits = df.loc[df.groupby('query_id')['score'].idxmax()]

    classifications = {}
    for _, row in best_hits.iterrows():
        # Extract sequence ID: "RL_seq_1" -> "seq_1"
        query_id = row['query_id']
        for prefix in [f'{blast_prefix}_', 'SFT_GRPO_']:
            if query_id.startswith(prefix):
                query_id = query_id[len(prefix):]
                break
        classifications[query_id] = classify_sequence(row['pct_identity'], row['query_coverage'])

    return classifications

def get_passed_ids(model):
    """Get set of sequence IDs that passed QC."""
    model_dir = MODEL_DIRS[model]
    passed_path = BASE_DIR / 'qc' / model_dir / 'passed.csv'
    if not passed_path.exists():
        return set()
    df = pd.read_csv(passed_path)
    return set(df['Plasmid_ID'].values)

def main():
    total_per_model = 100
    results = []

    print("Calculating actual QC + Novelty numbers:\n")

    for model in MODEL_ORDER:
        passed_ids = get_passed_ids(model)
        blast_class = get_blast_classifications(model)

        # Count sequences in each category
        n_passed = len(passed_ids)
        n_failed = total_per_model - n_passed

        # For passed sequences, check their novelty status
        passed_novel = 0
        passed_similar = 0
        passed_exists = 0
        passed_not_blasted = 0

        for seq_id in passed_ids:
            if seq_id in blast_class:
                cat = blast_class[seq_id]
                if cat == 'Novel':
                    passed_novel += 1
                elif cat == 'Similar':
                    passed_similar += 1
                else:
                    passed_exists += 1
            else:
                passed_not_blasted += 1

        # Also count BLASTed sequences that didn't pass QC
        blasted_ids = set(blast_class.keys())
        blasted_not_passed = blasted_ids - passed_ids

        print(f"{model}:")
        print(f"  Total: {total_per_model}")
        print(f"  Passed QC: {n_passed}")
        print(f"  Failed QC: {n_failed}")
        print(f"  Passed & BLASTed: {n_passed - passed_not_blasted}")
        print(f"    - Novel: {passed_novel}")
        print(f"    - Similar: {passed_similar}")
        print(f"    - Exists: {passed_exists}")
        print(f"  Passed but not BLASTed: {passed_not_blasted}")
        print(f"  BLASTed but not passed: {len(blasted_not_passed)}")
        print()

        results.append({
            'Model': model,
            'Total': total_per_model,
            'Passed QC': n_passed,
            'Failed QC': n_failed,
            'Passed & Novel': passed_novel,
            'Passed & Similar': passed_similar,
            'Passed & Exists': passed_exists,
            'Passed Not BLASTed': passed_not_blasted
        })

    df = pd.DataFrame(results).set_index('Model').reindex(MODEL_ORDER)
    df.to_csv(PUB_DIR / 'qc_novelty_actual.csv')
    print("Summary:")
    print(df[['Passed QC', 'Passed & Novel', 'Passed & Similar', 'Passed Not BLASTed']])

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.6

    colors = {
        'Failed QC': '#EF4444',
        'Passed & Exists': '#DC2626',
        'Passed & Similar': '#F59E0B',
        'Passed & Novel': '#22C55E',
        'Passed Not BLASTed': '#94A3B8'
    }

    bottom = np.zeros(len(MODEL_ORDER))
    for category in ['Failed QC', 'Passed & Similar', 'Passed & Novel', 'Passed Not BLASTed']:
        if category in df.columns:
            values = df[category].values
            bars = ax.bar(x, values, width, bottom=bottom, label=category,
                          color=colors[category], edgecolor='white', linewidth=1)
            for i, (val, bot) in enumerate(zip(values, bottom)):
                if val > 3:
                    ax.text(x[i], bot + val/2, str(int(val)),
                            ha='center', va='center', fontsize=11, fontweight='bold',
                            color='white' if 'Novel' not in category else 'black')
            bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Sequences (out of 100)', fontsize=12)
    ax.set_title('QC Pass Rate and NCBI Novelty (Actual)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig21_qc_novelty_actual.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig21_qc_novelty_actual.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: fig21_qc_novelty_actual.png/pdf")

if __name__ == '__main__':
    main()
