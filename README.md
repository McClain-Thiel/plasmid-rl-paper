# Plasmid-RL: Modeling Plasmid DNA using Reinforcement Learning

This repository contains the code, data, and manuscript for the ICML 2025 submission on using reinforcement learning to generate biologically valid plasmid DNA sequences.

## Repository Structure

```
plasmid_llm/
├── main.tex                    # ICML paper manuscript
├── plasmid_references.bib      # Bibliography
├── main.pdf                    # Compiled paper
│
├── code/
│   ├── figures/                # Figure generation scripts
│   │   ├── generate_figures.py           # Main figures (ICML-compliant)
│   │   ├── generate_additional_figures.py # Additional analysis figures
│   │   ├── create_distribution_grid_v2.py # Distribution grid with MFE
│   │   ├── create_exact_qc_novelty.py    # QC + novelty chart
│   │   └── generate_novelty_chart.py     # NCBI novelty classification
│   │
│   └── analysis/               # Analysis scripts
│       ├── analyze.py          # Core analysis (MinHash diversity, QC)
│       └── run_experiment_sweep.py # Full experiment pipeline
│
├── data/
│   ├── generations/            # Generated sequences (100 per model)
│   │   ├── Base/              # Pretrained model outputs
│   │   ├── SFT/               # Supervised fine-tuned outputs
│   │   ├── RL/                # RL (GRPO) outputs
│   │   └── SFT_GRPO/          # SFT+RL outputs
│   │
│   ├── qc/                     # Quality control results
│   │   ├── {model}/passed.csv  # Sequences passing QC
│   │   ├── {model}/repeats.csv # Repeat analysis
│   │   └── {model}/qc_summary.csv
│   │
│   ├── analysis/               # Benchmark results
│   │   ├── model_comparison_summary.csv
│   │   ├── completion_benchmark.csv
│   │   └── surprisal_benchmark.csv
│   │
│   ├── ncbi_blast/             # NCBI BLAST novelty results
│   │   └── {model}_blast_results.tsv
│   │
│   └── reference_plasmids/     # Real plasmids for comparison
│       ├── pUC19.fasta, pET-28a.fasta, ...
│
├── graphics/                   # Publication figures (PNG + PDF)
│
└── styles/                     # LaTeX style files (ICML 2025)
```

## Key Results

| Model  | QC Pass Rate | Diversity | Novelty Rate* |
|--------|--------------|-----------|---------------|
| Base   | 6%           | 0.93      | N/A           |
| SFT    | 11%          | 0.89      | 100%          |
| RL     | 97%          | 0.39      | 87%           |
| SFT+RL | 95%          | 0.45      | 96%           |

*Novelty rate among BLASTed QC-passing sequences

## Reproducing Results

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy ViennaRNA sourmash
```

### Generate Figures

```bash
cd code/figures

# Main publication figures
python generate_figures.py

# Distribution grid with MFE (several hours)
python create_distribution_grid_v2.py

# QC + Novelty analysis
python create_exact_qc_novelty.py
```

Note: Scripts use hardcoded paths that need to be updated to point to `../data/` for the repo structure.

### Compile Paper

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with latexmk:
```bash
latexmk -pdf main.tex
```

## Methods

### Models
- **Base**: PlasmidGPT pretrained on plasmid sequences
- **SFT**: Supervised fine-tuning on ~15k curated E. coli plasmids
- **RL**: Group Relative Policy Optimization (GRPO) with biological reward
- **SFT+RL**: Combined SFT then RL approach

### Quality Control Criteria
- Exactly 1 origin of replication (ORI)
- 1-2 antimicrobial resistance (AMR) genes
- No internal repeats >50 bp

### Novelty Classification (NCBI BLAST)
- **Exists**: identity ≥99% AND coverage ≥95%
- **Similar**: identity ≥95% AND coverage ≥80%
- **Novel**: anything else

### Diversity Metric
MinHash diversity using sourmash (k=21, n=1000):
```
D = 1 - mean(pairwise Jaccard similarities)
```

## Reference Plasmids

The following common lab plasmids are used for distribution comparison and benchmarking:

| Plasmid | Size | Description |
|---------|------|-------------|
| pUC19 | 2,686 bp | High-copy cloning vector |
| pBluescript | 2,961 bp | Cloning vector |
| pBR322 | 4,361 bp | Classic cloning vector |
| pACYC184 | 4,245 bp | Low-copy vector |
| pBAD24 | 4,542 bp | Arabinose-inducible expression |
| pEGFP | 4,733 bp | GFP expression vector |
| pGEX-4T-1 | 4,969 bp | GST fusion expression |
| pET-28a | 5,369 bp | T7 expression vector |
| pcDNA3 | 5,428 bp | Mammalian expression |
| px330 | 8,074 bp | CRISPR/Cas9 vector |

## Citation

```bibtex
@inproceedings{plasmidrl2025,
  title={Modeling Plasmid DNA using Reinforcement Learning},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
# plasmid-rl-paper
