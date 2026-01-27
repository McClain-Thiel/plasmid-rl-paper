#!/usr/bin/env python3
"""
Publication figure generation for plasmid language model comparison.
ICML-ready figures with seaborn white theme.
3 Models Only: Base, SFT, RL (no SFT+RL)
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
BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
PUB_DIR = BASE_DIR / 'publication'
PUB_DIR.mkdir(exist_ok=True)

# Model order for all plots - 3 models only
MODEL_ORDER = ['Base', 'SFT', 'RL']
MODEL_COLORS = {'Base': '#2E4057', 'SFT': '#8B5CF6', 'RL': '#E11D48'}

# Mapping from directory names to display names
DIR_TO_NAME = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL'}
NAME_TO_DIR = {v: k for k, v in DIR_TO_NAME.items()}

# Figure settings - ICML style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.family'] = 'sans-serif'

sns.set_theme(style="white")

def load_all_data():
    """Load and combine all data from Base, SFT, RL models only."""
    all_data = []

    for model_dir in ['Base', 'SFT', 'RL']:  # Only 3 models
        model_name = DIR_TO_NAME[model_dir]

        # Load outputs (sequences and prompts)
        outputs_path = BASE_DIR / 'generations' / model_dir / 'outputs.csv'
        outputs = pd.read_csv(outputs_path)
        outputs['model'] = model_name
        outputs['prompt_type'] = outputs['prompt'].apply(
            lambda x: 'GFP' if len(x) > 10 else 'ATG'
        )

        # Load repeats (has sequence length)
        repeats_path = BASE_DIR / 'qc' / model_dir / 'repeats.csv'
        repeats = pd.read_csv(repeats_path)
        repeats = repeats.rename(columns={'plasmid_id': 'id'})

        # Load QC summary (ORI and AMR counts)
        qc_path = BASE_DIR / 'qc' / model_dir / 'qc_summary.csv'
        qc = pd.read_csv(qc_path)
        qc = qc.rename(columns={'sample': 'id'})

        # Load passed sequences
        passed_path = BASE_DIR / 'qc' / model_dir / 'passed.csv'
        passed = pd.read_csv(passed_path)
        passed_ids = set(passed['Plasmid_ID'].values)

        # Merge data
        merged = outputs.merge(repeats[['id', 'seq_length', 'circular', 'longest_len', 'longest_fraction']],
                               on='id', how='left')
        merged = merged.merge(qc[['id', 'n_ori_kept', 'n_amr']], on='id', how='left')
        merged['passed'] = merged['id'].isin(passed_ids)

        all_data.append(merged)

    return pd.concat(all_data, ignore_index=True)

def compute_gc_content(seq):
    """Compute GC content of a sequence."""
    if pd.isna(seq) or len(seq) == 0:
        return np.nan
    seq = seq.upper()
    gc = sum(1 for c in seq if c in 'GC')
    return gc / len(seq) * 100

def add_sequence_metrics(df):
    """Add computed metrics to dataframe."""
    print("Computing GC content...")
    df['gc_content'] = df['full'].apply(compute_gc_content)

    print("Computing log10 length...")
    df['log_length'] = np.log10(df['seq_length'].replace(0, np.nan))

    return df

# ============== FIGURE GENERATION FUNCTIONS ==============

def plot_pass_rate_overall(df):
    """Bar chart of pass rates by model (both prompts combined)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    pass_rates = df.groupby('model')['passed'].mean() * 100
    pass_rates = pass_rates.reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, pass_rates.values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1.2, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, pass_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=13)
    ax.set_title('Plasmid QC Pass Rate by Model', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig01_pass_rate_overall.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig01_pass_rate_overall.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig01_pass_rate_overall")

def plot_pass_rate_by_prompt(df):
    """Bar chart of pass rates by model and prompt type."""
    fig, ax = plt.subplots(figsize=(9, 5))

    pass_rates = df.groupby(['model', 'prompt_type'])['passed'].mean() * 100
    pass_rates = pass_rates.unstack()
    pass_rates = pass_rates.reindex(MODEL_ORDER)

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    bars1 = ax.bar(x - width/2, pass_rates['ATG'], width, label='ATG Prompt',
                   color='#3B82F6', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, pass_rates['GFP'], width, label='GFP Prompt',
                   color='#F59E0B', edgecolor='black', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=13)
    ax.set_title('Plasmid QC Pass Rate by Model and Prompt Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig02_pass_rate_by_prompt.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig02_pass_rate_by_prompt.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig02_pass_rate_by_prompt")

def plot_diversity_overall():
    """Bar chart of diversity by model."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Load diversity from summary and filter to 3 models
    summary = pd.read_csv(BASE_DIR / 'analysis' / 'model_comparison_summary.csv')
    summary['Model'] = summary['Model'].replace('GRPO', 'RL')
    summary = summary[summary['Model'].isin(['Base', 'SFT', 'RL'])]
    summary = summary.set_index('Model').reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, summary['Diversity'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1.2, width=0.6)

    # Add value labels
    for bar, val in zip(bars, summary['Diversity'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Self-Diversity (Mash Distance)', fontsize=13)
    ax.set_title('Sequence Diversity by Model', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig03_diversity_overall.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig03_diversity_overall.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig03_diversity_overall")

def plot_pass_rate_vs_diversity():
    """Scatter plot showing pass rate vs diversity trade-off."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Load summary data and filter to 3 models
    summary = pd.read_csv(BASE_DIR / 'analysis' / 'model_comparison_summary.csv')
    summary['Model'] = summary['Model'].replace('GRPO', 'RL')
    summary = summary[summary['Model'].isin(['Base', 'SFT', 'RL'])]

    for _, row in summary.iterrows():
        model = row['Model']
        ax.scatter(row['Diversity'], row['PassRate'],
                   s=250, c=MODEL_COLORS[model], edgecolors='black', linewidth=2,
                   label=model, zorder=5)
        ax.annotate(model, (row['Diversity'], row['PassRate']),
                    xytext=(12, 5), textcoords='offset points', fontsize=12, fontweight='bold')

    ax.set_xlabel('Self-Diversity (Mash Distance)', fontsize=13)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=13)
    ax.set_title('Pass Rate vs. Diversity Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig04_pass_vs_diversity.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig04_pass_vs_diversity.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig04_pass_vs_diversity")

def plot_length_distribution(df):
    """Violin/box plot of sequence lengths by model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter to reasonable lengths (> 100 bp)
    df_filt = df[df['seq_length'] > 100].copy()
    df_filt['log_length'] = np.log10(df_filt['seq_length'])

    # Create ordered categorical
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.violinplot(data=df_filt, x='model', y='log_length', palette=palette,
                   order=MODEL_ORDER, ax=ax, inner='box')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Sequence Length (log$_{10}$ bp)', fontsize=13)
    ax.set_title('Distribution of Generated Plasmid Lengths', fontsize=14, fontweight='bold')

    # Add reference lines for typical plasmid sizes
    ax.axhline(y=np.log10(3000), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=np.log10(10000), color='gray', linestyle=':', alpha=0.5)
    ax.text(2.3, np.log10(3000)+0.05, '3 kb', fontsize=9, color='gray')
    ax.text(2.3, np.log10(10000)+0.05, '10 kb', fontsize=9, color='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig05_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig05_length_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig05_length_distribution")

def plot_gc_distribution(df):
    """Distribution of GC content by model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter to sequences with valid GC
    df_filt = df[df['gc_content'].notna() & (df['seq_length'] > 100)].copy()
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.violinplot(data=df_filt, x='model', y='gc_content', palette=palette,
                   order=MODEL_ORDER, ax=ax, inner='box')

    # Add reference line for typical plasmid GC (around 50%)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.text(2.3, 51, '50%', fontsize=9, color='gray')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('GC Content (%)', fontsize=13)
    ax.set_title('Distribution of GC Content', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig06_gc_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig06_gc_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig06_gc_distribution")

def plot_repeat_fraction(df):
    """Distribution of longest repeat fraction by model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    df_filt = df[df['longest_fraction'].notna() & (df['seq_length'] > 100)].copy()
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.boxplot(data=df_filt, x='model', y='longest_fraction', palette=palette,
                order=MODEL_ORDER, ax=ax)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Longest Repeat Fraction', fontsize=13)
    ax.set_title('Distribution of Repeat Content', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig07_repeat_fraction.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig07_repeat_fraction.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig07_repeat_fraction")

def plot_completion_benchmark():
    """Plot completion benchmark results."""
    fig, ax = plt.subplots(figsize=(7, 5))

    completion = pd.read_csv(BASE_DIR / 'analysis' / 'completion_benchmark.csv')
    # Filter to 3 models only
    completion = completion[completion['Model'].isin(['Base', 'SFT', 'RL'])]

    # Aggregate by model
    model_means = completion.groupby('Model')['AvgLogProb'].mean()
    model_stds = completion.groupby('Model')['AvgLogProb'].std()
    model_means = model_means.reindex(MODEL_ORDER)
    model_stds = model_stds.reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, model_means.values,
                  yerr=model_stds.values, capsize=5,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1.2, width=0.6)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Mean Log Probability', fontsize=13)
    ax.set_title('Plasmid Completion Benchmark', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Note: higher (less negative) is better
    ax.text(0.02, 0.98, 'Higher = Better', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic')

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig08_completion_benchmark.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig08_completion_benchmark.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig08_completion_benchmark")

def plot_surprisal_benchmark():
    """Plot surprisal benchmark results."""
    fig, ax = plt.subplots(figsize=(7, 5))

    surprisal = pd.read_csv(BASE_DIR / 'analysis' / 'surprisal_benchmark.csv')
    # Filter to 3 models only
    surprisal = surprisal[surprisal['Model'].isin(['Base', 'SFT', 'RL'])]

    # Aggregate by model
    model_means = surprisal.groupby('Model')['MeanLogProb'].mean()
    model_stds = surprisal.groupby('Model')['MeanLogProb'].std()
    model_means = model_means.reindex(MODEL_ORDER)
    model_stds = model_stds.reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, model_means.values,
                  yerr=model_stds.values, capsize=5,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1.2, width=0.6)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Mean Log Probability at Functional Sites', fontsize=13)
    ax.set_title('Surprisal at Promoter-CDS Junctions', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.02, 0.98, 'Higher = Better', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic')

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig09_surprisal_benchmark.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig09_surprisal_benchmark.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig09_surprisal_benchmark")

def plot_novelty_chart():
    """Plot BLAST novelty results."""
    # Load summary and filter
    summary = pd.read_csv(BASE_DIR / 'analysis' / 'model_comparison_summary.csv')
    summary['Model'] = summary['Model'].replace('GRPO', 'RL')
    summary = summary[summary['Model'].isin(['Base', 'SFT', 'RL'])]
    summary = summary.set_index('Model').reindex(MODEL_ORDER)

    if 'Novelty_Pct' not in summary.columns:
        print("Skipping plot_novelty_chart: 'Novelty_Pct' column missing.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(MODEL_ORDER, summary['Novelty_Pct'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1.2, width=0.6)

    # Add value labels
    for bar, val in zip(bars, summary['Novelty_Pct'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Novelty Rate (%)', fontsize=13)
    ax.set_title('BLAST Novelty Rate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig10_novelty_rate.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig10_novelty_rate.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig10_novelty_rate")

def plot_ori_amr_counts(df):
    """Plot ORI and AMR gene counts by model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    df_filt = df[df['seq_length'] > 100].copy()
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    # ORI counts
    ax = axes[0]
    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.boxplot(data=df_filt, x='model', y='n_ori_kept', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of ORIs', fontsize=12)
    ax.set_title('A) Origin of Replication Count', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # AMR counts
    ax = axes[1]
    sns.boxplot(data=df_filt, x='model', y='n_amr', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of AMR Genes', fontsize=12)
    ax.set_title('B) Antibiotic Resistance Gene Count', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig11_ori_amr_counts.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig11_ori_amr_counts.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig11_ori_amr_counts")

def plot_combined_summary():
    """Create a combined 2x2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Load summary data
    summary = pd.read_csv(BASE_DIR / 'analysis' / 'model_comparison_summary.csv')
    summary['Model'] = summary['Model'].replace('GRPO', 'RL')
    summary = summary[summary['Model'].isin(['Base', 'SFT', 'RL'])]
    summary = summary.set_index('Model').reindex(MODEL_ORDER)

    # Panel A: Pass Rate
    ax = axes[0, 0]
    bars = ax.bar(MODEL_ORDER, summary['PassRate'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1, width=0.6)
    for bar, val in zip(bars, summary['PassRate'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('QC Pass Rate (%)', fontsize=11)
    ax.set_title('A) QC Pass Rate', fontsize=12, fontweight='bold', loc='left')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Diversity
    ax = axes[0, 1]
    bars = ax.bar(MODEL_ORDER, summary['Diversity'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1, width=0.6)
    for bar, val in zip(bars, summary['Diversity'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Self-Diversity', fontsize=11)
    ax.set_title('B) Sequence Diversity', fontsize=12, fontweight='bold', loc='left')
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Trade-off scatter
    ax = axes[1, 0]
    for model in MODEL_ORDER:
        row = summary.loc[model]
        ax.scatter(row['Diversity'], row['PassRate'],
                   s=200, c=MODEL_COLORS[model], edgecolors='black', linewidth=2,
                   label=model, zorder=5)
        ax.annotate(model, (row['Diversity'], row['PassRate']),
                    xytext=(10, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    ax.set_xlabel('Self-Diversity', fontsize=11)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=11)
    ax.set_title('C) Pass Rate vs. Diversity', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel D: Novelty Rate
    ax = axes[1, 1]
    if 'Novelty_Pct' in summary.columns:
        bars = ax.bar(MODEL_ORDER, summary['Novelty_Pct'].values,
                      color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                      edgecolor='black', linewidth=1, width=0.6)
        for bar, val in zip(bars, summary['Novelty_Pct'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('BLAST Novelty Rate (%)', fontsize=11)
        ax.set_title('D) Novelty Rate', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylim(0, 110)
    else:
        ax.text(0.5, 0.5, 'Novelty Data Missing', ha='center', va='center')
        ax.set_title('D) Novelty Rate (Missing)', fontsize=12, fontweight='bold', loc='left')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_DIR / 'fig00_summary_panel.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / 'fig00_summary_panel.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fig00_summary_panel")

# ============== MAIN ==============

def main():
    print("=" * 60)
    print("Generating Publication Figures (3 Models: Base, SFT, RL)")
    print("=" * 60)

    # Load all data
    print("\nLoading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} sequences across {df['model'].nunique()} models")

    # Add computed metrics
    print("\nComputing sequence metrics...")
    df = add_sequence_metrics(df)

    # Save combined data
    df.to_csv(PUB_DIR / 'combined_data_3models.csv', index=False)
    print(f"Saved combined data to {PUB_DIR / 'combined_data_3models.csv'}")

    # Generate figures
    print("\nGenerating figures...")
    print("-" * 40)

    plot_combined_summary()
    plot_pass_rate_overall(df)
    plot_pass_rate_by_prompt(df)
    plot_diversity_overall()
    plot_pass_rate_vs_diversity()
    plot_length_distribution(df)
    plot_gc_distribution(df)
    plot_repeat_fraction(df)
    plot_completion_benchmark()
    plot_surprisal_benchmark()
    plot_novelty_chart()
    plot_ori_amr_counts(df)

    print("\n" + "=" * 60)
    print("All figures saved to:", PUB_DIR)
    print("=" * 60)

if __name__ == '__main__':
    main()
