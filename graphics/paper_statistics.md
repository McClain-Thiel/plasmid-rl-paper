# Paper Statistics
**Results Directory:** `results_temp_1.1`
---
## 1. QC Pass Rates
- **Base:** 5/100 (5.0%)
- **SFT:** 10/100 (10.0%)
- **RL:** 77/100 (77.0%)

## 2. Held-Out Continuation Task (Log-Probability)
| Model | Mean Log-Prob | Std | N |
|-------|---------------|-----|---|
| Base | -12.4492 | 6.1440 | 85 |
| SFT | -12.4492 | 6.1440 | 85 |
| RL | -10.9660 | 2.7415 | 85 |

### Paired T-Tests (Completion)
- Base vs SFT: t=3.945, p=1.65e-04 *** (n=85 pairs)
- Base vs RL: t=-2.474, p=1.54e-02 * (n=85 pairs)
- SFT vs RL: t=-2.474, p=1.54e-02 * (n=85 pairs)

### Alignment Tax Analysis
- **Base vs RL (paired t-test)**
  - N pairs: 85
  - Base mean: -12.4492
  - RL mean: -10.9660
  - Difference (RL - Base): +1.4831
  - t-statistic: -2.4741
  - p-value: 1.54e-02 *
  - Cohen's d: 0.268 (small effect)
  - **Interpretation: No alignment tax** (RL performs better)

## 3. Surprisal Benchmark
| Model | Mean Log-Prob | Std | N |
|-------|---------------|-----|---|
| Base | -13.8384 | 2.5110 | 28 |
| SFT | -13.8384 | 2.5110 | 28 |
| RL | -11.6637 | 1.8210 | 28 |

### Paired T-Tests (Surprisal)
- Base vs SFT: t=7.408, p=5.72e-08 *** (n=28 pairs)
- Base vs RL: t=-5.128, p=2.16e-05 *** (n=28 pairs)
- SFT vs RL: t=-5.128, p=2.16e-05 *** (n=28 pairs)

## 4. Distribution Metrics
### 3-mer KL Divergence from Real Plasmids
| Model | KL(Model||Real) | Mean JS Divergence |
|-------|-----------------|--------------------|
| Base | 0.0142 | 0.1037 |
| SFT | 0.0084 | 0.1074 |
| RL | 0.0106 | 0.0866 |

### GC Content
| Model | Mean GC | Std |
|-------|---------|-----|
| Real | 0.5172 | 0.0291 |
| Base | 0.4779 | 0.0336 |
| SFT | 0.4953 | 0.0501 |
| RL | 0.5185 | 0.0293 |

### Sequence Length
| Model | Mean Length | Std | Median |
|-------|-------------|-----|--------|
| Real | 7273 | 2924 | 6690 |
| Base | 7482 | 3829 | 7168 |
| SFT | 5812 | 4202 | 4409 |
| RL | 7107 | 1489 | 6668 |

### MFE Density (Thermodynamic Stability)
| Model | Mean MFE/nt | Std |
|-------|-------------|-----|
| Real | -0.3643 | 0.0267 |
| Base | -0.3597 | 0.0652 |
| SFT | -0.3577 | 0.0728 |
| RL | -0.3622 | 0.0338 |

## 5. Diversity Metrics
| Model | Pass Rate (%) | Diversity Score |
|-------|---------------|----------------|
| Base | 5.0 | 0.9150 |
| SFT | 10.0 | 0.8964 |
| GRPO | 77.0 | 0.5878 |

## 6. Sample Counts
| Category | Count |
|----------|-------|
| Real | 254 |
| Base | 53 |
| SFT | 100 |
| RL | 100 |
| **Total** | 507 |

