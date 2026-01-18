#!/usr/bin/env python3
"""Parse BLAST text output and classify novelty."""

import re
import sys

def parse_blast_file(filepath):
    """Parse BLAST text output and extract best hit for each query."""
    with open(filepath) as f:
        content = f.read()

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
        # Look for "Identities = X/Y (Z%)"
        identity_matches = re.findall(r'Identities = (\d+)/(\d+) \((\d+)%\)', query_block)

        if identity_matches:
            # Get first (best) hit
            matches, total, pct = identity_matches[0]
            identity_pct = int(pct)

            # Calculate coverage as matched bases / query length
            coverage = int(matches) / length * 100 if length > 0 else 0

            # Get hit name
            hit_match = re.search(r'>([A-Z]{1,2}\d+\.\d+) (.+?)(?:\n|Length=)', query_block)
            hit_name = hit_match.group(1) if hit_match else 'Unknown'
            hit_desc = hit_match.group(2).strip() if hit_match else ''

            # Classify
            # Exists: ≥99% identity AND ≥95% coverage
            # Similar: ≥95% identity AND ≥80% coverage
            # Novel: otherwise
            if identity_pct >= 99 and coverage >= 95:
                classification = 'Exists'
            elif identity_pct >= 95 and coverage >= 80:
                classification = 'Similar'
            else:
                classification = 'Novel'

            results.append({
                'query': query_name,
                'length': length,
                'best_hit': f"{hit_name} {hit_desc[:50]}",
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

def main():
    print("=" * 80)
    print("BASE MODEL BLAST RESULTS")
    print("=" * 80)
    base_results = parse_blast_file('Base_blast_full.txt')
    for r in base_results:
        print(f"\n{r['query']} (len={r['length']})")
        if r['best_hit']:
            print(f"  Best hit: {r['best_hit']}")
            print(f"  Identity: {r['identity']}%, Coverage: {r['coverage']}%")
        else:
            print(f"  No significant hits")
        print(f"  Classification: {r['classification']}")

    base_novel = sum(1 for r in base_results if r['classification'] == 'Novel')
    base_similar = sum(1 for r in base_results if r['classification'] == 'Similar')
    base_exists = sum(1 for r in base_results if r['classification'] == 'Exists')
    print(f"\nBase Summary: {base_novel} Novel, {base_similar} Similar, {base_exists} Exists out of {len(base_results)}")
    print(f"Base Novelty Rate: {(base_novel + base_similar) / len(base_results) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("SFT MODEL BLAST RESULTS")
    print("=" * 80)
    sft_results = parse_blast_file('SFT_blast_full.txt')
    for r in sft_results:
        print(f"\n{r['query']} (len={r['length']})")
        if r['best_hit']:
            print(f"  Best hit: {r['best_hit']}")
            print(f"  Identity: {r['identity']}%, Coverage: {r['coverage']}%")
        else:
            print(f"  No significant hits")
        print(f"  Classification: {r['classification']}")

    sft_novel = sum(1 for r in sft_results if r['classification'] == 'Novel')
    sft_similar = sum(1 for r in sft_results if r['classification'] == 'Similar')
    sft_exists = sum(1 for r in sft_results if r['classification'] == 'Exists')
    print(f"\nSFT Summary: {sft_novel} Novel, {sft_similar} Similar, {sft_exists} Exists out of {len(sft_results)}")
    print(f"SFT Novelty Rate: {(sft_novel + sft_similar) / len(sft_results) * 100:.1f}%")

    # Write CSV summary
    with open('blast_summary.csv', 'w') as f:
        f.write('model,query,length,best_hit,identity_pct,coverage_pct,classification\n')
        for r in base_results:
            hit = r['best_hit'].replace(',', ';') if r['best_hit'] else 'None'
            f.write(f"Base,{r['query']},{r['length']},{hit},{r['identity']},{r['coverage']},{r['classification']}\n")
        for r in sft_results:
            hit = r['best_hit'].replace(',', ';') if r['best_hit'] else 'None'
            f.write(f"SFT,{r['query']},{r['length']},{hit},{r['identity']},{r['coverage']},{r['classification']}\n")

    print("\nWrote blast_summary.csv")

if __name__ == '__main__':
    main()
