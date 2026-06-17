#!/usr/bin/env python3
"""
Pass@k evaluation using vLLM + verl reward functions.

Evaluates a merged HF checkpoint on multiple test sets,
computing pass@1, pass@4, pass@8 using the unbiased Codex estimator.

Usage:
  python tools/eval_passk_final.py \
      --model_dir /path/to/merged/hf \
      --output results/baseline_step1500.json \
      --tp 8 --n 8 --temperature 0.6
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# pass@k unbiased estimator (Codex paper)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_stats(per_problem: list[dict], k_values: list[int]) -> dict:
    stats = {}
    for k in k_values:
        valid = [r for r in per_problem if r["n"] >= k]
        if not valid:
            stats[f"pass@{k}"] = 0.0
            continue
        stats[f"pass@{k}"] = sum(pass_at_k(r["n"], r["c"], k) for r in valid) / len(valid)
    return stats


# ---------------------------------------------------------------------------
# Scoring via verl reward functions
# ---------------------------------------------------------------------------

def score_one(data_source: str, solution_str: str, ground_truth: str) -> float:
    """Score a single response using verl's reward functions."""
    from verl.utils.reward_score import default_compute_score
    try:
        result = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
        )
        if isinstance(result, dict):
            return float(result.get("score", result.get("acc", 0.0)))
        return float(result)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompts_from_parquet(data_file: str, tokenizer) -> tuple[list[str], list[dict]]:
    """
    Load parquet, apply chat template, return (prompt_strings, metadata).
    metadata[i] = {"data_source": ..., "ground_truth": ..., "problem": ...}
    """
    df = pd.read_parquet(data_file)
    prompts = []
    meta = []

    for _, row in df.iterrows():
        # Messages already formatted in training style
        messages = list(row["prompt"])  # list of {"role": ..., "content": ...}

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)

        ground_truth = row["reward_model"]["ground_truth"]
        problem = messages[0]["content"] if messages else ""
        meta.append({
            "data_source": row["data_source"],
            "ground_truth": ground_truth,
            "problem": problem[:200],
        })

    return prompts, meta


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def run_inference(model_dir: str, prompts: list[str], n: int,
                  temperature: float, max_tokens: int,
                  tp: int, gpu_mem: float) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    print(f"[vLLM] Loading: {model_dir}")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_tokens,
        trust_remote_code=True,
    )

    # For n=1 or greedy, use temperature=0
    actual_temp = temperature if n > 1 else 0.0
    params = SamplingParams(
        n=n,
        temperature=actual_temp,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    print(f"[vLLM] Generating {len(prompts)} prompts × {n} samples (temp={actual_temp}) ...")
    outputs = llm.generate(prompts, params)

    return [[o.text for o in out.outputs] for out in outputs]


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model_dir: str,
    data_file: str,
    n: int,
    temperature: float,
    k_values: list[int],
    max_tokens: int,
    tp: int,
    gpu_mem: float,
    tokenizer,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {data_file}")
    print(f"{'='*60}")

    prompts, meta = build_prompts_from_parquet(data_file, tokenizer)
    print(f"  {len(prompts)} problems, data_source={meta[0]['data_source']}")

    all_gens = run_inference(model_dir, prompts, n, temperature, max_tokens, tp, gpu_mem)

    per_problem = []
    for i, (m, gens) in enumerate(zip(meta, all_gens)):
        correct_flags = [
            score_one(m["data_source"], gen, m["ground_truth"]) >= 0.5
            for gen in gens
        ]
        c = sum(correct_flags)
        per_problem.append({
            "idx": i,
            "problem": m["problem"],
            "ground_truth": m["ground_truth"],
            "data_source": m["data_source"],
            "n": len(gens),
            "c": c,
            "acc": c / len(gens),
        })

    stats = compute_pass_stats(per_problem, k_values)
    never = sum(1 for r in per_problem if r["c"] == 0)
    always = sum(1 for r in per_problem if r["c"] == n)

    print(f"\n  Results ({len(prompts)} problems, n={n}, T={temperature}):")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v:.4f} ({v:.2%})")
    print(f"  never correct: {never}/{len(prompts)} ({never/len(prompts):.1%})")
    print(f"  always correct: {always}/{len(prompts)} ({always/len(prompts):.1%})")

    return {
        "data_file": data_file,
        "n_problems": len(prompts),
        "data_source": meta[0]["data_source"],
        "n": n,
        "temperature": temperature,
        "pass_at_k": stats,
        "never_correct": never,
        "always_correct": always,
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "math500":  "~/data/processed/math500_full/test.parquet",
    "gsm8k":    "~/data/processed/gsm8k/test.parquet",
    "aime2024": "~/data/processed/aime2024_full/test.parquet",
    "aime2025": "~/data/processed/aime2025_full/test.parquet",
    "amc":      "~/data/processed/amc/test.parquet",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="which datasets to evaluate")
    args = parser.parse_args()

    model_dir = str(Path(args.model_dir).expanduser())

    # Load tokenizer once
    from transformers import AutoTokenizer
    print(f"[init] Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    all_results = {
        "model_dir": model_dir,
        "n": args.n,
        "temperature": args.temperature,
        "k_values": args.k,
        "datasets": {},
    }

    # Summary table
    summary = {}

    for ds_name in args.datasets:
        data_file = str(Path(DATASETS[ds_name]).expanduser())
        result = evaluate_dataset(
            model_dir=model_dir,
            data_file=data_file,
            n=args.n,
            temperature=args.temperature,
            k_values=args.k,
            max_tokens=args.max_tokens,
            tp=args.tp,
            gpu_mem=args.gpu_mem,
            tokenizer=tokenizer,
        )
        all_results["datasets"][ds_name] = result
        summary[ds_name] = result["pass_at_k"]

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY — {model_dir}")
    print(f"{'='*70}")
    header = f"{'Dataset':<12}" + "".join(f"  pass@{k:<6}" for k in sorted(args.k))
    print(header)
    print("-" * len(header))
    for ds_name in args.datasets:
        row = f"{ds_name:<12}"
        for k in sorted(args.k):
            v = summary[ds_name].get(f"pass@{k}", 0.0)
            row += f"  {v:.4f}      "
        print(row)
    print(f"{'='*70}\n")

    # Save
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
