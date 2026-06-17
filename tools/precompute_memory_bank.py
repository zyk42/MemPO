#!/usr/bin/env python3
"""
Precompute EMA-GRPO memory bank initialization using the base model.

Runs n rollouts per prompt on the base (untuned) model, computes batch_mean
for each prompt, and writes a memory bank JSON compatible with PromptMemoryBank.load().

Usage:
    python tools/precompute_memory_bank.py \
        --model_dir /mnt/lisiqi23/models/Qwen3-1.7B \
        --train_file ~/data/processed/math500_full/test.parquet \
        --output /tmp/memory_bank_init.json \
        --n 8 \
        --max_tokens 12288 \
        --temperature 0.6 \
        --tp 1 \
        --gpu_mem 0.6 \
        --batch_size 32

Output JSON format (compatible with PromptMemoryBank.load()):
    {
        "bank": {"<index>": [mean, count], ...},
        "initial_acc": {"<index>": mean, ...}
    }
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reward scoring (reuse verl's reward function)
# ---------------------------------------------------------------------------

def score_response(data_source: str, solution_str: str, ground_truth: str) -> float:
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
# Prompt rendering (same as eval_passk_final.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant. Let's think step by step and output the final answer within \\boxed{}."

def render_prompt(row, tokenizer) -> str:
    messages = []
    prompt_data = row["prompt"]
    if isinstance(prompt_data, list):
        messages = prompt_data
    elif isinstance(prompt_data, str):
        messages = [{"role": "user", "content": prompt_data}]
    else:
        messages = [{"role": "user", "content": str(prompt_data)}]

    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to base HF model")
    parser.add_argument("--train_file", required=True, help="Training parquet file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--n", type=int, default=8, help="Rollouts per prompt")
    parser.add_argument("--max_tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.6, help="vLLM gpu_memory_utilization")
    parser.add_argument("--batch_size", type=int, default=32, help="Prompts per vLLM call")
    args = parser.parse_args()

    model_dir = str(Path(args.model_dir).expanduser())
    train_file = str(Path(args.train_file).expanduser())
    output_path = str(Path(args.output).expanduser())

    # ── Load dataset ──
    df = pd.read_parquet(train_file)
    print(f"[precompute] Loaded {len(df)} prompts from {train_file}")

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # ── Launch vLLM ──
    from vllm import LLM, SamplingParams
    print(f"[precompute] Loading model: {model_dir} (tp={args.tp})")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_tokens,
        trust_remote_code=True,
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    # ── Build prompts ──
    prompts = []
    indices = []
    data_sources = []
    ground_truths = []

    for _, row in df.iterrows():
        extra = row["extra_info"]
        if isinstance(extra, dict):
            idx = int(extra["index"])
        else:
            idx = int(extra)
        indices.append(idx)
        data_sources.append(str(row.get("data_source", "math")))
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, dict):
            gt = reward_model.get("ground_truth", "")
        else:
            gt = str(reward_model)
        ground_truths.append(gt)
        prompts.append(render_prompt(row, tokenizer))

    # ── Generate in batches ──
    bank = {}       # index_str -> [mean, count]
    initial_acc = {}  # index_str -> mean

    n_prompts = len(prompts)
    n_batches = math.ceil(n_prompts / args.batch_size)
    print(f"[precompute] Generating {args.n} rollouts × {n_prompts} prompts "
          f"in {n_batches} batches of {args.batch_size}")

    for b in range(n_batches):
        start = b * args.batch_size
        end = min(start + args.batch_size, n_prompts)
        batch_prompts = prompts[start:end]
        batch_indices = indices[start:end]
        batch_sources = data_sources[start:end]
        batch_gts = ground_truths[start:end]

        outputs = llm.generate(batch_prompts, sampling_params)

        for i, (out, idx, src, gt) in enumerate(
            zip(outputs, batch_indices, batch_sources, batch_gts)
        ):
            scores = []
            for completion in out.outputs:
                s = score_response(src, completion.text, gt)
                scores.append(s)

            batch_mean = float(np.mean(scores)) if scores else 0.0
            key = str(idx)
            bank[key] = [batch_mean, 1]   # count=1 (cold-start initialized)
            initial_acc[key] = batch_mean

        print(f"[precompute] Batch {b+1}/{n_batches} done "
              f"(prompts {start+1}–{end}, "
              f"mean_acc={np.mean([bank[str(i)][0] for i in batch_indices]):.3f})")

    # ── Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    result = {"bank": bank, "initial_acc": initial_acc}
    with open(output_path, "w") as f:
        json.dump(result, f)

    overall_mean = np.mean([v[0] for v in bank.values()])
    print(f"\n[precompute] Done. {len(bank)} prompts, overall μ₀={overall_mean:.3f}")
    print(f"[precompute] Saved to: {output_path}")


if __name__ == "__main__":
    main()
