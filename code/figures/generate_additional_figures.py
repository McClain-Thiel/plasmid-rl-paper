#!/usr/bin/env python3
"""
Additional publication figures - diversity by prompt, ORF analysis, k-mer analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path('/Users/mcclainthiel/Downloads/results_Exp14_Aggressive')
PUB_DIR = BASE_DIR / 'publication'

# Model order and colors
MODEL_ORDER = ['Base', 'SFT', 'RL', 'SFT+RL']
MODEL_COLORS = {'Base': '#2E4057', 'SFT': '#8B5CF6', 'RL': '#E11D48', 'SFT+RL': '#F97316'}
DIR_TO_NAME = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL', 'SFT_GRPO': 'SFT+RL'}

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_theme(style="white")

def load_combined_data():
    """Load the combined data from earlier processing."""
    return pd.read_csv(PUB_DIR / 'combined_data.csv')

def compute_diversity_by_prompt(df):
    """
    Compute pairwise sequence diversity within each model/prompt group.
    Uses simple edit distance ratio as a proxy for diversity.
    """
    from difflib import SequenceMatcher

    results = []

    for model in MODEL_ORDER:
        model_dir = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL', 'SFT+RL': 'SFT_GRPO'}[model]
        for prompt_type in ['ATG', 'GFP']:
            subset = df[(df['model'] == model) & (df['prompt_type'] == prompt_type)]
            seqs = subset['full'].dropna().values

            if len(seqs) < 2:
                continue

            # Sample pairwise comparisons (for efficiency)
            n_comparisons = min(100, len(seqs) * (len(seqs) - 1) // 2)
            similarities = []

            np.random.seed(42)
            for _ in range(n_comparisons):
                i, j = np.random.choice(len(seqs), 2, replace=False)
                # Use shorter sequences for speed
                s1 = seqs[i][:5000] if len(seqs[i]) > 5000 else seqs[i]
                s2 = seqs[j][:5000] if len(seqs[j]) > 5000 else seqs[j]
                ratio = SequenceMatcher(None, s1, s2).ratio()
                similarities.append(ratio)

            # Diversity = 1 - mean similarity
            diversity = 1 - np.mean(similarities)
            results.append({
                'Model': model,
                'Prompt': prompt_type,
                'Diversity': diversity
            })

    return pd.DataFrame(results)

def extract_orf_data():
    """Extract ORF information from GFF files."""
    all_orfs = []

    for model_dir in ['Base', 'SFT', 'RL', 'SFT_GRPO']:
        model_name = DIR_TO_NAME[model_dir]
        qc_dir = BASE_DIR / 'qc' / model_dir / 'individual_sequences_qc'

        for seq_dir in qc_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            seq_id = seq_dir.name
            gff_files = list(seq_dir.glob('*.gff'))

            if not gff_files:
                continue

            gff_file = gff_files[0]

            # Parse GFF
            orfs = []
            seq_length = 0

            with open(gff_file) as f:
                for line in f:
                    if line.startswith('# Sequence Data'):
                        # Extract sequence length
                        for part in line.split(';'):
                            if 'seqlen=' in part:
                                seq_length = int(part.split('=')[1])
                    elif not line.startswith('#') and '\t' in line:
                        parts = line.strip().split('\t')
                        if len(parts) >= 5 and parts[2] == 'CDS':
                            start = int(parts[3])
                            end = int(parts[4])
                            orf_len = (end - start + 1) // 3  # amino acids
                            orfs.append(orf_len)

            # Calculate metrics
            if orfs:
                all_orfs.append({
                    'seq_id': seq_id,
                    'model': model_name,
                    'seq_length': seq_length,
                    'n_orfs': len(orfs),
                    'n_orfs_100aa': sum(1 for o in orfs if o >= 100),
                    'longest_orf': max(orfs),
                    'total_coding': sum(orfs) * 3,
                    'coding_fraction': sum(orfs) * 3 / seq_length if seq_length > 0 else 0
                })

    return pd.DataFrame(all_orfs)

def compute_kmer_js_divergence(df, k=3):
    """Compute k-mer JS divergence between generated sequences and a reference distribution."""

    def get_kmer_freq(seq, k):
        if pd.isna(seq) or len(seq) < k:
            return {}
        seq = seq.upper()
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        counts = Counter(kmers)
        total = sum(counts.values())
        return {kmer: count/total for kmer, count in counts.items()}

    def js_divergence(p, q):
        all_keys = set(p.keys()) | set(q.keys())
        p_arr = np.array([p.get(k, 0) for k in all_keys])
        q_arr = np.array([q.get(k, 0) for k in all_keys])
        eps = 1e-10
        p_arr = p_arr + eps
        q_arr = q_arr + eps
        p_arr = p_arr / p_arr.sum()
        q_arr = q_arr / q_arr.sum()
        m = (p_arr + q_arr) / 2
        return 0.5 * (np.sum(p_arr * np.log(p_arr / m)) + np.sum(q_arr * np.log(q_arr / m)))

    # Compute reference distribution (from all sequences combined)
    all_seqs = df[df['seq_length'] > 1000]['full'].dropna()
    ref_freq = Counter()
    for seq in all_seqs:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            ref_freq[seq[i:i+k]] += 1
    total = sum(ref_freq.values())
    ref_freq = {kmer: count/total for kmer, count in ref_freq.items()}

    # Compute JS divergence for each sequence
    results = []
    for _, row in df[df['seq_length'] > 1000].iterrows():
        seq_freq = get_kmer_freq(row['full'], k)
        if seq_freq:
            js = js_divergence(seq_freq, ref_freq)
            results.append({
                'model': row['model'],
                'seq_id': row['id'],
                'js_divergence': js
            })

    return pd.DataFrame(results)

# ============== FIGURE FUNCTIONS ==============

def plot_diversity_by_prompt():
    """Bar chart of diversity by model and prompt type."""
    # Load pre-computed summary (using mash-based diversity from the original)
    # Since we don't have per-prompt mash diversity, we'll compute approximate diversity

    df = load_combined_data()

    print("Computing diversity by prompt (this may take a moment)...")
    div_df = compute_diversity_by_prompt(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = div_df.pivot(index='Model', columns='Prompt', values='Diversity')
    pivot = pivot.reindex(MODEL_ORDER)

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    bars1 = ax.bar(x - width/2, pivot['ATG'], width, label='ATG Prompt',
                   color='#3B82F6', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, pivot['GFP'], width, label='GFP Prompt',
                   color='#F59E0B', edgecolor='black', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Sequence Diversity', fontsize=12)
    ax.set_title('Sequence Diversity by Model and Prompt Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig13_diversity_by_prompt.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig13_diversity_by_prompt.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig13_diversity_by_prompt")

def plot_orf_distributions():
    """Plot ORF count and length distributions."""
    print("Extracting ORF data from GFF files...")
    orf_df = extract_orf_data()
    orf_df['model'] = pd.Categorical(orf_df['model'], categories=MODEL_ORDER, ordered=True)

    # Filter to sequences with reasonable length
    orf_df = orf_df[orf_df['seq_length'] > 1000]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]

    # Panel A: Number of ORFs >= 100aa
    ax = axes[0]
    sns.boxplot(data=orf_df, x='model', y='n_orfs_100aa', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of ORFs (≥100 aa)', fontsize=12)
    ax.set_title('A) ORF Count (≥100 aa)', fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Longest ORF
    ax = axes[1]
    sns.boxplot(data=orf_df, x='model', y='longest_orf', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Longest ORF (amino acids)', fontsize=12)
    ax.set_title('B) Longest ORF Length', fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Coding fraction
    ax = axes[2]
    sns.boxplot(data=orf_df, x='model', y='coding_fraction', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Coding Fraction', fontsize=12)
    ax.set_title('C) Coding Sequence Fraction', fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig14_orf_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig14_orf_analysis.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig14_orf_analysis")

    # Save ORF data
    orf_df.to_csv(PUB_DIR / 'orf_metrics.csv', index=False)
    print("Saved: orf_metrics.csv")

def plot_kmer_analysis():
    """Plot 3-mer JS divergence by model."""
    df = load_combined_data()

    print("Computing 3-mer JS divergence...")
    js_df = compute_kmer_js_divergence(df, k=3)
    js_df['model'] = pd.Categorical(js_df['model'], categories=MODEL_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]

    sns.violinplot(data=js_df, x='model', y='js_divergence', palette=palette,
                   order=MODEL_ORDER, ax=ax, inner='box')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('3-mer JS Divergence', fontsize=12)
    ax.set_title('3-mer Compositional Divergence from Reference', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig15_kmer_divergence.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig15_kmer_divergence.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig15_kmer_divergence")

def plot_length_by_prompt():
    """Plot length distribution by model and prompt."""
    df = load_combined_data()
    df = df[df['seq_length'] > 100].copy()
    df['log_length'] = np.log10(df['seq_length'])
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, prompt in enumerate(['ATG', 'GFP']):
        ax = axes[idx]
        subset = df[df['prompt_type'] == prompt]
        palette = [MODEL_COLORS[m] for m in MODEL_ORDER]

        sns.violinplot(data=subset, x='model', y='log_length', palette=palette,
                       order=MODEL_ORDER, ax=ax, inner='box')

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Sequence Length (log₁₀ bp)', fontsize=12)
        ax.set_title(f'{"A" if idx == 0 else "B"}) {prompt} Prompt',
                     fontsize=12, fontweight='bold', loc='left')
        ax.axhline(y=np.log10(3000), color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=np.log10(10000), color='gray', linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig16_length_by_prompt.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig16_length_by_prompt.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig16_length_by_prompt")

def plot_qc_breakdown():
    """Plot detailed QC pass/fail breakdown by failure reason."""
    failure_data = []

    for model_dir in ['Base', 'SFT', 'RL', 'SFT_GRPO']:
        model_name = DIR_TO_NAME[model_dir]
        failed_path = BASE_DIR / 'qc' / model_dir / 'failed.csv'
        passed_path = BASE_DIR / 'qc' / model_dir / 'passed.csv'

        failed = pd.read_csv(failed_path)
        passed = pd.read_csv(passed_path)

        n_passed = len(passed)
        n_failed = len(failed)
        n_total = n_passed + n_failed

        # Parse failure reasons
        no_ori = sum(failed['reason failed'].str.contains('No ORI', na=False))
        no_amr = sum(failed['reason failed'].str.contains('No ARG', na=False))
        repeat = sum(failed['reason failed'].str.contains('repeat', na=False))
        multi_ori = sum(failed['reason failed'].str.contains('ORI.*count.*outside', na=False))
        multi_amr = sum(failed['reason failed'].str.contains('ARG.*count.*outside', na=False))

        failure_data.append({
            'Model': model_name,
            'Passed': n_passed,
            'No ORI': no_ori,
            'No AMR': no_amr,
            'Excess Repeats': repeat,
            'Multiple ORIs': multi_ori,
            'Multiple AMRs': multi_amr
        })

    fail_df = pd.DataFrame(failure_data)
    fail_df = fail_df.set_index('Model').reindex(MODEL_ORDER)

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Passed', 'No ORI', 'No AMR', 'Excess Repeats']
    colors = ['#22C55E', '#EF4444', '#F97316', '#8B5CF6']

    bottom = np.zeros(len(MODEL_ORDER))
    for cat, color in zip(categories, colors):
        if cat in fail_df.columns:
            values = fail_df[cat].values
            ax.bar(MODEL_ORDER, values, bottom=bottom, label=cat, color=color, edgecolor='white')
            bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Sequences', fontsize=12)
    ax.set_title('QC Outcome Breakdown', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig17_qc_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig17_qc_breakdown.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig17_qc_breakdown")

def main():
    print("=" * 60)
    print("Generating Additional Publication Figures")
    print("=" * 60)

    plot_diversity_by_prompt()
    plot_orf_distributions()
    plot_kmer_analysis()
    plot_length_by_prompt()
    plot_qc_breakdown()

    print("\n" + "=" * 60)
    print("Additional figures saved to:", PUB_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
