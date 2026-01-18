#!/usr/bin/env python3
"""
Create distribution grid with MFE (thermodynamic stability), not repeat content.
All samples combined per model (no prompt breakdown).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import jensenshannon
import RNA
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('/Users/mcclainthiel/Downloads/results_Exp14_Aggressive')
PUB_DIR = BASE_DIR / 'publication'
REAL_DIR = Path('/Users/mcclainthiel/Projects/PhD/analysis2/assets/annotations')

MODEL_ORDER = ['Real', 'Base', 'SFT', 'RL', 'SFT+RL']
MODEL_COLORS = {
    'Real': '#22C55E', 'Base': '#2E4057', 'SFT': '#8B5CF6',
    'RL': '#E11D48', 'SFT+RL': '#F97316'
}
DIR_TO_NAME = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL', 'SFT_GRPO': 'SFT+RL'}

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
sns.set_theme(style="white")

def compute_gc(seq):
    if pd.isna(seq) or len(seq) == 0:
        return np.nan
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq)

def compute_mfe_density(seq):
    """Compute MFE density (MFE / length) for circular DNA."""
    if pd.isna(seq) or len(seq) < 100:
        return np.nan
    try:
        seq = seq.upper().replace('N', 'A')  # Replace N with A
        md = RNA.md()
        md.circ = 1  # Circular
        fc = RNA.fold_compound(seq, md)
        _, mfe = fc.mfe()
        return mfe / len(seq)
    except:
        return np.nan

def kmer_distribution(seq, k=3):
    if pd.isna(seq) or len(seq) < k:
        return {}
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    return {kmer: count/total for kmer, count in counts.items()}

def js_divergence_to_ref(seq, ref_dist, k=3):
    seq_dist = kmer_distribution(seq, k)
    if not seq_dist:
        return np.nan
    all_keys = set(seq_dist.keys()) | set(ref_dist.keys())
    p = np.array([seq_dist.get(k, 0) for k in all_keys])
    q = np.array([ref_dist.get(k, 0) for k in all_keys])
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(jensenshannon(p, q, base=2.0))

def load_data():
    """Load all sequences."""
    all_data = []

    # Load real plasmids
    for f in REAL_DIR.glob('*.fasta'):
        if 'cassette' in f.name.lower():
            continue
        with open(f) as fh:
            seq = ''.join(line.strip() for line in fh if not line.startswith('>'))
            if seq and len(seq) > 1000:
                all_data.append({'model': 'Real', 'full': seq.upper(), 'seq_length': len(seq)})

    # Load generated sequences
    for model_dir in ['Base', 'SFT', 'RL', 'SFT_GRPO']:
        model_name = DIR_TO_NAME[model_dir]
        outputs_path = BASE_DIR / 'generations' / model_dir / 'outputs.csv'
        if outputs_path.exists():
            df = pd.read_csv(outputs_path)
            for _, row in df.iterrows():
                if pd.notna(row['full']) and len(row['full']) > 500:
                    all_data.append({
                        'model': model_name,
                        'full': row['full'].upper(),
                        'seq_length': len(row['full'])
                    })

    return pd.DataFrame(all_data)

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} sequences: {df['model'].value_counts().to_dict()}")

    print("Computing GC content...")
    df['gc'] = df['full'].apply(compute_gc)

    print("Computing MFE density (this may take a while)...")
    total = len(df)
    mfe_values = []
    for i, seq in enumerate(df['full']):
        if i % 10 == 0:
            print(f"  MFE progress: {i}/{total} ({100*i/total:.0f}%)", flush=True)
        mfe_values.append(compute_mfe_density(seq))
    df['mfe_density'] = mfe_values
    print(f"  MFE progress: {total}/{total} (100%)")

    print("Computing 3-mer JS divergence...")
    real_seqs = df[df['model'] == 'Real']['full'].dropna()
    ref_concat = ''.join(real_seqs.values) if len(real_seqs) > 0 else ''.join(df['full'].dropna().values)
    ref_dist = kmer_distribution(ref_concat, k=3)
    df['js_3mer'] = df['full'].apply(lambda x: js_divergence_to_ref(x, ref_dist, k=3))

    df['log_length'] = np.log10(df['seq_length'])
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)

    # Save metrics
    df.to_csv(PUB_DIR / 'computed_metrics_with_mfe.csv', index=False)
    print("Saved computed metrics")

    # Create figure
    print("Creating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    order = [m for m in MODEL_ORDER if m in df['model'].unique()]
    palette = [MODEL_COLORS[m] for m in order]

    # A) Sequence Length
    ax = axes[0, 0]
    sns.violinplot(data=df, x='model', y='seq_length', order=order, palette=palette, ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('Length (bp)')
    ax.set_title('A) Sequence Length', fontweight='bold', loc='left')
    ax.axhline(y=3000, color='gray', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # B) 3-mer JS Divergence
    ax = axes[0, 1]
    sns.violinplot(data=df, x='model', y='js_3mer', order=order, palette=palette, ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('JS Divergence (3-mer)')
    ax.set_title('B) 3-mer Compositional Divergence', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # C) MFE Density (Thermodynamic Stability)
    ax = axes[1, 0]
    sns.violinplot(data=df, x='model', y='mfe_density', order=order, palette=palette, ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('MFE Density (kcal/mol/nt)')
    ax.set_title('C) Thermodynamic Stability', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # D) GC Content
    ax = axes[1, 1]
    sns.violinplot(data=df, x='model', y='gc', order=order, palette=palette, ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('GC Content')
    ax.set_title('D) GC Content', fontweight='bold', loc='left')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig_distribution_grid_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(PUB_DIR / 'fig_distribution_grid_v2.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: fig_distribution_grid_v2.png/pdf")

if __name__ == '__main__':
    main()
