#!/usr/bin/env python3
"""Parse all BLAST results and create comprehensive summary."""

import re
import os

def parse_blast_file(filepath):
    """Parse BLAST text output and extract best hit for each query."""
    with open(filepath) as f:
        content = f.read()

    # Check for errors
    if 'Error:' in content and 'CPU usage limit' in content:
        print(f"  WARNING: {filepath} has CPU error, skipping")
        return []

    # Split by Query=
    queries = re.split(r'\nQuery= ', content)

    results = []
    for query_block in queries[1:]:  # Skip header
        lines = query_block.strip().split('\n')
        query_name = lines[0].strip()

        # Find length
        length_match = re.search(r'Length=(\d+)', query_block)
        length = int(length_match.group(1)) if length_match else 0

        # Check for "No significant similarity found"
        if 'No significant similarity found' in query_block:
            results.append({
                'query': query_name,
                'length': length,
                'best_hit': None,
                'identity': 0,
                'coverage': 0,
                'classification': 'Novel'
            })
            continue

        # Find best hit identity from first alignment
        identity_matches = re.findall(r'Identities = (\d+)/(\d+) \((\d+)%\)', query_block)

        if identity_matches:
            matches, total, pct = identity_matches[0]
            identity_pct = int(pct)
            coverage = int(matches) / length * 100 if length > 0 else 0

            hit_match = re.search(r'>([A-Z]{1,2}\d+\.\d+) (.+?)(?:\n|Length=)', query_block)
            hit_name = hit_match.group(1) if hit_match else 'Unknown'
            hit_desc = hit_match.group(2).strip() if hit_match else ''

            # Classify
            if identity_pct >= 99 and coverage >= 95:
                classification = 'Exists'
            elif identity_pct >= 95 and coverage >= 80:
                classification = 'Similar'
            else:
                classification = 'Novel'

            results.append({
                'query': query_name,
                'length': length,
                'best_hit': f"{hit_name} {hit_desc[:40]}",
                'identity': identity_pct,
                'coverage': round(coverage, 1),
                'classification': classification
            })
        else:
            results.append({
                'query': query_name,
                'length': length,
                'best_hit': None,
                'identity': 0,
                'coverage': 0,
                'classification': 'Novel'
            })

    return results

# Parse all files
all_results = {
    'Base': [],
    'SFT': [],
    'RL': [],
    'SFT_GRPO': []
}

# Original passed QC sequences
print("Parsing original BLAST results (passed QC)...")
all_results['Base'].extend(parse_blast_file('Base_blast_full.txt'))
all_results['SFT'].extend(parse_blast_file('SFT_blast_full.txt'))

# Additional batches (didn't pass QC)
print("Parsing Base batches...")
for i in range(1, 5):
    fname = f'Base_batch{i}_blast.txt'
    if os.path.exists(fname):
        all_results['Base'].extend(parse_blast_file(fname))

print("Parsing SFT batches...")
for i in range(1, 4):
    fname = f'SFT_batch{i}_blast.txt'
    if os.path.exists(fname):
        all_results['SFT'].extend(parse_blast_file(fname))

print("Parsing RL additional...")
if os.path.exists('RL_additional_blast.txt'):
    all_results['RL'].extend(parse_blast_file('RL_additional_blast.txt'))

print("Parsing SFT_GRPO additional...")
if os.path.exists('SFT_GRPO_additional_blast.txt'):
    all_results['SFT_GRPO'].extend(parse_blast_file('SFT_GRPO_additional_blast.txt'))

# Count passed QC for each model
passed_qc = {
    'Base': 6,
    'SFT': 11,
    'RL': 97,
    'SFT_GRPO': 95
}

# Previously BLASTed from RL and SFT_GRPO (passed QC only)
# RL: 23 BLASTed, 20 Novel, 3 Exists
# SFT_GRPO: 24 BLASTed, 23 Novel, 1 Exists
prev_blasted = {
    'RL': {'blasted': 23, 'novel': 20, 'similar': 0, 'exists': 3},
    'SFT_GRPO': {'blasted': 24, 'novel': 23, 'similar': 0, 'exists': 1}
}

print("\n" + "="*80)
print("COMPREHENSIVE BLAST SUMMARY")
print("="*80)

# Summary statistics
for model in ['Base', 'SFT', 'RL', 'SFT_GRPO']:
    results = all_results[model]

    # For RL and SFT_GRPO, add previous results
    if model in prev_blasted:
        prev = prev_blasted[model]
        total_blasted = prev['blasted'] + len(results)
        # Count from new results
        new_novel = sum(1 for r in results if r['classification'] == 'Novel')
        new_similar = sum(1 for r in results if r['classification'] == 'Similar')
        new_exists = sum(1 for r in results if r['classification'] == 'Exists')

        novel = prev['novel'] + new_novel
        similar = prev['similar'] + new_similar
        exists = prev['exists'] + new_exists

        # Passed QC and novel - only from the original BLASTed (which were all passed QC)
        # The new additional RL/SFT_GRPO sequences were also from passed QC pool
        passed_qc_novel = prev['novel'] + new_novel  # all RL/SFT_GRPO BLASTed passed QC
    else:
        total_blasted = len(results)
        novel = sum(1 for r in results if r['classification'] == 'Novel')
        similar = sum(1 for r in results if r['classification'] == 'Similar')
        exists = sum(1 for r in results if r['classification'] == 'Exists')

        # For Base and SFT: first 6/11 passed QC, rest didn't
        if model == 'Base':
            passed_qc_novel = sum(1 for r in results[:6] if r['classification'] == 'Novel')
        elif model == 'SFT':
            passed_qc_novel = sum(1 for r in results[:11] if r['classification'] == 'Novel')

    novelty_pct = (novel + similar) / total_blasted * 100 if total_blasted > 0 else 0
    passed_qc_novel_pct = passed_qc_novel / 100 * 100  # percentage of total 100 sequences

    print(f"\n{model}:")
    print(f"  BLASTed: {total_blasted}")
    print(f"  Novel: {novel}, Similar: {similar}, Exists: {exists}")
    print(f"  Novelty Rate: {novelty_pct:.1f}%")
    print(f"  Passed QC: {passed_qc[model]}")
    print(f"  Passed QC & Novel: {passed_qc_novel} ({passed_qc_novel_pct:.1f}% of total)")

    all_results[model + '_stats'] = {
        'blasted': total_blasted,
        'novel': novel,
        'similar': similar,
        'exists': exists,
        'novelty_pct': novelty_pct,
        'passed_qc': passed_qc[model],
        'passed_qc_novel': passed_qc_novel,
        'passed_qc_novel_pct': passed_qc_novel_pct
    }

# Write updated CSV
print("\n" + "="*80)
print("Writing model_comparison_summary.csv...")
diversity = {'Base': 0.926, 'SFT': 0.886, 'RL': 0.391, 'SFT_GRPO': 0.453}
pass_rate = {'Base': 6.0, 'SFT': 11.0, 'RL': 97.0, 'SFT_GRPO': 95.0}

with open('/Users/mcclainthiel/Downloads/plasmid_llm/data/analysis/model_comparison_summary.csv', 'w') as f:
    f.write('Model,PassRate,PassedQC,Diversity,BLASTed,Novel,Similar,Exists,Novelty_Pct,PassedQC_and_Novel,PassedQC_and_Novel_Pct\n')
    for model in ['Base', 'SFT', 'RL', 'SFT_GRPO']:
        stats = all_results[model + '_stats']
        f.write(f"{model},{pass_rate[model]},{passed_qc[model]},{diversity[model]:.3f},"
                f"{stats['blasted']},{stats['novel']},{stats['similar']},{stats['exists']},"
                f"{stats['novelty_pct']:.1f},{stats['passed_qc_novel']},{stats['passed_qc_novel_pct']:.1f}\n")

print("Done!")

# Also write detailed CSV
with open('blast_all_results.csv', 'w') as f:
    f.write('model,query,length,best_hit,identity_pct,coverage_pct,classification,passed_qc\n')
    for model in ['Base', 'SFT', 'RL', 'SFT_GRPO']:
        for i, r in enumerate(all_results[model]):
            # Determine if passed QC
            if model == 'Base':
                pqc = 'yes' if i < 6 else 'no'
            elif model == 'SFT':
                pqc = 'yes' if i < 11 else 'no'
            else:
                pqc = 'yes'  # RL and SFT_GRPO additional were all passed QC

            hit = r['best_hit'].replace(',', ';') if r['best_hit'] else 'None'
            f.write(f"{model},{r['query']},{r['length']},{hit},{r['identity']},{r['coverage']},{r['classification']},{pqc}\n")

print("Wrote blast_all_results.csv")
