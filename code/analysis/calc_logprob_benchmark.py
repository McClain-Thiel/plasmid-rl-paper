#!/usr/bin/env python3
"""
Calculate log probability of held-out continuations for all 4 models.
Run on GPU server: python calc_logprob_benchmark.py
"""

import torch
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
from tqdm import tqdm

# Model paths from HuggingFace
MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT": "UCL-CSSB/PlasmidGPT-SFT",
    "RL": "UCL-CSSB/PlasmidGPT-GRPO",
    "SFT_GRPO": "McClain/PlasmidGPT-RL"
}

# Parameters for completion windows
WINDOW_PREFIX = 400  # bp context
WINDOW_TARGET = 100  # bp to predict
CUT_STRIDE = 300     # sliding window stride


def get_model_logprobs(model, tokenizer, seq, device):
    """Calculate per-token log probabilities for a sequence."""
    enc = tokenizer(seq, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)

    # Handle out-of-vocab tokens
    if (input_ids >= model.config.vocab_size).any():
        input_ids[input_ids >= model.config.vocab_size] = 0

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits

    # Log softmax to get log probabilities
    lps = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get log prob of actual next tokens
    target_ids = input_ids[:, 1:]
    lps = lps[:, :-1, :]
    gathered = lps.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    return gathered.cpu().numpy()[0]


def create_completion_windows(real_dir):
    """Create windows from real plasmid sequences."""
    windows = []

    for fname in os.listdir(real_dir):
        if not fname.endswith('.fasta'):
            continue

        fpath = os.path.join(real_dir, fname)
        try:
            seq = str(SeqIO.read(fpath, "fasta").seq).upper()
            # Filter to valid DNA characters
            seq = "".join([c for c in seq if c in "ATGC"])

            if len(seq) < WINDOW_PREFIX + WINDOW_TARGET + 10:
                continue

            # Create sliding windows
            for start in range(0, len(seq) - (WINDOW_PREFIX + WINDOW_TARGET), CUT_STRIDE):
                prefix = seq[start : start + WINDOW_PREFIX]
                target = seq[start + WINDOW_PREFIX : start + WINDOW_PREFIX + WINDOW_TARGET]
                windows.append((fname, start, prefix, target))
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

    return windows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU will be slow!")

    # Find real plasmids directory
    possible_paths = [
        "../data/reference_plasmids",
        "../../data/reference_plasmids",
        "/home/ubuntu/plasmid_llm/data/reference_plasmids",
        "./reference_plasmids"
    ]

    real_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            real_dir = p
            break

    if real_dir is None:
        print("ERROR: Could not find reference_plasmids directory!")
        print("Please ensure the directory exists at one of:", possible_paths)
        return

    print(f"Using reference plasmids from: {real_dir}")

    # Create completion windows
    print("Creating completion windows...")
    windows = create_completion_windows(real_dir)
    print(f"Created {len(windows)} windows from {len(set(w[0] for w in windows))} plasmids")

    # Run benchmark for each model
    results = []

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Loading {model_name}: {model_path}")
        print('='*50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device).eval()

            for (plasmid, start, prefix, target) in tqdm(windows, desc=f"{model_name}"):
                full_seq = prefix + target
                trace = get_model_logprobs(model, tokenizer, full_seq, device)

                # Find where target starts in tokens
                prefix_tokens = tokenizer(prefix, return_tensors="pt").input_ids.shape[1]
                target_start_idx = prefix_tokens - 1

                if target_start_idx < len(trace):
                    avg_logprob = float(np.mean(trace[target_start_idx:]))
                    results.append({
                        "Model": model_name,
                        "Plasmid": plasmid,
                        "Start": start,
                        "AvgLogProb": avg_logprob
                    })

            # Free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR loading {model_name}: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    output_path = "completion_benchmark_new.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = df.groupby('Model')['AvgLogProb'].agg(['mean', 'std', 'count'])
    print(summary.round(3))

    # Quick check if models are identical
    print("\n" + "="*60)
    print("CHECKING FOR IDENTICAL MODELS")
    print("="*60)
    pivot = df.pivot_table(index=['Plasmid', 'Start'], columns='Model', values='AvgLogProb')
    for m1 in MODELS.keys():
        for m2 in MODELS.keys():
            if m1 < m2 and m1 in pivot.columns and m2 in pivot.columns:
                diff = (pivot[m1] - pivot[m2]).abs().max()
                print(f"{m1} vs {m2}: max diff = {diff:.6f}")


if __name__ == "__main__":
    main()
