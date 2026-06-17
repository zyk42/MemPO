#!/usr/bin/env python3
"""转换 + 评测 GSM8K 实验的 checkpoint（pass@1/4/8）。

用法：
  python3 tools/eval_gsm8k.py
"""
import gc, json, math, re, subprocess, sys
from pathlib import Path

MOUNT  = Path("/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research")
DATA   = Path.home() / "data/processed/gsm8k_test_split/test_no_think.parquet"
MERGE  = Path("/root/verl/experiments/grpo-forgetting-research/tools/merge_lora_direct.py")
OUT_DIR = MOUNT / "eval_results_gsm8k"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    ("baseline_step1000", MOUNT / "checkpoints_gsm8k_baseline_grpo/global_step_1000"),
    ("baseline_step1240", MOUNT / "checkpoints_gsm8k_baseline_grpo/global_step_1240"),
    ("phase3_step1000",   MOUNT / "checkpoints_gsm8k_phase3_alpha0.9/global_step_1000"),
    ("phase3_step1240",   MOUNT / "checkpoints_gsm8k_phase3_alpha0.9/global_step_1240"),
]

# ---------------------------------------------------------------------------
def convert(ckpt_dir: Path, hf_dir: Path):
    if (hf_dir / "model.safetensors").exists():
        print(f"[SKIP convert] {hf_dir.name} already exists")
        return
    print(f"[convert] {ckpt_dir} → {hf_dir}")
    hf_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(MERGE),
        "--ckpt_dir", str(ckpt_dir / "actor"),
        "--output_dir", str(hf_dir),
        "--lora_alpha", "32",
        "--lora_r", "64",
    ], check=True)

# ---------------------------------------------------------------------------
def load_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
    samples = []
    for _, row in df.iterrows():
        messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
        gt = row["reward_model"]["ground_truth"]
        samples.append({"messages": messages, "ground_truth": str(gt)})
    return samples

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    return matches[-1].strip() if matches else None

def normalize_num(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return s.lower()

def is_correct(pred, gt):
    p, g = normalize_num(extract_boxed(pred) if pred else None), normalize_num(gt)
    if p is None or g is None:
        return False
    if isinstance(p, float) and isinstance(g, float):
        return abs(p - g) < 1e-6
    return str(p) == str(g)

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def run_eval(model_path, samples, n=8, temperature=0.6, tp=4):
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    print(f"\n[eval] loading {model_path.name}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=str(model_path), tensor_parallel_size=tp,
               gpu_memory_utilization=0.85, max_model_len=4096,
               trust_remote_code=True)
    prompts = [tok.apply_chat_template(s["messages"], tokenize=False,
                add_generation_prompt=True) for s in samples]
    outputs = llm.generate(prompts,
                SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=2048))
    per_problem = []
    for i, (out, samp) in enumerate(zip(outputs, samples)):
        gens = [o.text for o in out.outputs]
        c = sum(is_correct(g, samp["ground_truth"]) for g in gens)
        per_problem.append({"idx": i, "gt": samp["ground_truth"], "n": n, "c": c})
    del llm; gc.collect(); torch.cuda.empty_cache()
    return per_problem

# ---------------------------------------------------------------------------
samples = load_parquet(DATA)
print(f"Test set: {len(samples)} problems")
summary = {}

for label, ckpt_dir in MODELS:
    hf_dir = MOUNT / f"hf_models_gsm8k_{label}"
    out_file = OUT_DIR / f"{label}.json"

    # 转换
    convert(ckpt_dir, hf_dir)

    # 评测
    if out_file.exists():
        print(f"[SKIP eval] {label}")
        with open(out_file) as f:
            summary[label] = json.load(f)["pass_at_k"]
        continue

    per_problem = run_eval(hf_dir, samples, n=8, tp=4)
    k_vals = [1, 4, 8]
    stats = {f"pass@{k}": sum(pass_at_k(r["n"], r["c"], k)
                               for r in per_problem) / len(per_problem)
             for k in k_vals}
    print(f"[result] {label}: {stats}")
    with open(out_file, "w") as f:
        json.dump({"model": label, "n": 8, "pass_at_k": stats,
                   "per_problem": per_problem}, f, indent=2)
    summary[label] = stats

# 打印汇总
print(f"\n{'='*60}")
print(f"  {'模型':<25}  {'pass@1':>7}  {'pass@4':>7}  {'pass@8':>7}")
print(f"  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*7}")
for label, stats in summary.items():
    print(f"  {label:<25}  {stats.get('pass@1',0):>7.4f}  "
          f"{stats.get('pass@4',0):>7.4f}  {stats.get('pass@8',0):>7.4f}")
print(f"{'='*60}")

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n结果保存到 {OUT_DIR / 'summary.json'}")
