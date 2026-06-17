#!/usr/bin/env python3
"""批量评测所有实验模型的 pass@1 / pass@4 / pass@8（n=8, temperature=0.6）。

用法：
  python tools/eval_pass8.py --split test
  python tools/eval_pass8.py --split both --tp 4
  python tools/eval_pass8.py --split both --tp 4 --models baseline_step480 phase2_g2.0_step480
"""

import argparse
import json
import math
import re
import gc
from pathlib import Path


MOUNT = Path("/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research")
DATA_ROOT = Path.home() / "data/processed/math500"

MODELS = [
    ("original",              Path.home() / "model/Qwen3-1.7B"),
    ("baseline_step400",      MOUNT / "hf_models_baseline_qwen3/global_step_400/hf"),
    ("baseline_step480",      MOUNT / "hf_models_baseline_qwen3/global_step_480/hf"),
    ("phase2_g2.0_step400",   MOUNT / "hf_models_phase2_alpha0.9_gamma2.0/global_step_400/hf"),
    ("phase2_g2.0_step480",   MOUNT / "hf_models_phase2_alpha0.9_gamma2.0/global_step_480/hf"),
    ("phase3_g0.0_step400",   MOUNT / "hf_models_phase3_filter_only_alpha0.9_gamma0.0/global_step_400/hf"),
    ("phase3_g0.0_step480",   MOUNT / "hf_models_phase3_filter_only_alpha0.9_gamma0.0/global_step_480/hf"),
    ("phase4_g0.5_step400",   MOUNT / "hf_models_phase4_alpha0.9_gamma0.5/global_step_400/hf"),
    ("phase4_g0.5_step480",   MOUNT / "hf_models_phase4_alpha0.9_gamma0.5/global_step_480/hf"),
    ("phase4_g1.0_step400",   MOUNT / "hf_models_phase4_alpha0.9_gamma1.0/global_step_400/hf"),
    ("phase4_g1.0_step480",   MOUNT / "hf_models_phase4_alpha0.9_gamma1.0/global_step_480/hf"),
    ("phase5_s1.0_step400",   MOUNT / "hf_models_phase5_soft_alpha0.9_soft1.0/global_step_400/hf"),
    ("phase5_s1.0_step480",   MOUNT / "hf_models_phase5_soft_alpha0.9_soft1.0/global_step_480/hf"),
    ("phase5_s0.5_step400",   MOUNT / "hf_models_phase5_soft_alpha0.9_soft0.5/global_step_400/hf"),
    ("phase5_s0.5_step480",   MOUNT / "hf_models_phase5_soft_alpha0.9_soft0.5/global_step_480/hf"),
    ("phase5_s2.0_step400",   MOUNT / "hf_models_phase5_soft_alpha0.9_soft2.0/global_step_400/hf"),
    ("phase5_s2.0_step480",   MOUNT / "hf_models_phase5_soft_alpha0.9_soft2.0/global_step_480/hf"),
    ("phase6_g0.5_step400",   MOUNT / "hf_models_phase6_reactive_alpha0.9_gamma0.5/global_step_400/hf"),
    ("phase6_g0.5_step480",   MOUNT / "hf_models_phase6_reactive_alpha0.9_gamma0.5/global_step_480/hf"),
    ("phase6_g1.0_step400",   MOUNT / "hf_models_phase6_reactive_alpha0.9_gamma1.0/global_step_400/hf"),
    ("phase6_g1.0_step480",   MOUNT / "hf_models_phase6_reactive_alpha0.9_gamma1.0/global_step_480/hf"),
    ("phase6_g2.0_step400",   MOUNT / "hf_models_phase6_reactive_alpha0.9_gamma2.0/global_step_400/hf"),
    ("phase6_g2.0_step480",   MOUNT / "hf_models_phase6_reactive_alpha0.9_gamma2.0/global_step_480/hf"),
    ("phase3_ext_step640",    MOUNT / "hf_models_phase3_filter_only_alpha0.9_gamma0.0/global_step_640/hf"),
    ("phase3_ext_step800",    MOUNT / "hf_models_phase3_filter_only_alpha0.9_gamma0.0/global_step_800/hf"),
    ("phase3_ext_step960",    MOUNT / "hf_models_phase3_filter_only_alpha0.9_gamma0.0/global_step_960/hf"),
    ("phase7_dg0.5_step400",  MOUNT / "hf_models_phase7_replay_alpha0.9_dgamma0.5/global_step_400/hf"),
    ("phase7_dg0.5_step480",  MOUNT / "hf_models_phase7_replay_alpha0.9_dgamma0.5/global_step_480/hf"),
    ("phase7_dg1.0_step400",  MOUNT / "hf_models_phase7_replay_alpha0.9_dgamma1.0/global_step_400/hf"),
    ("phase7_dg1.0_step480",  MOUNT / "hf_models_phase7_replay_alpha0.9_dgamma1.0/global_step_480/hf"),
    ("phase7_dg2.0_step400",  MOUNT / "hf_models_phase7_replay_alpha0.9_dgamma2.0/global_step_400/hf"),
    ("phase7_dg2.0_step480",  MOUNT / "hf_models_phase7_replay_alpha0.9_dgamma2.0/global_step_480/hf"),
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_parquet(data_file: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(data_file)
    samples = []
    for _, row in df.iterrows():
        messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
        gt = row["reward_model"]["ground_truth"]
        samples.append({"messages": messages, "ground_truth": str(gt)})
    return samples


# ---------------------------------------------------------------------------
# 答案提取 / 正确性判断
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def normalize(ans: str) -> str:
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("$", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("{", "").replace("}", "")
    return ans


def is_correct(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    return normalize(pred) == normalize(gt)


# ---------------------------------------------------------------------------
# pass@k（Codex 无偏估计）
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k(per_problem: list[dict], n: int, k_values: list[int]) -> dict:
    result = {}
    for k in k_values:
        vals = [pass_at_k(n, r["c"], k) for r in per_problem]
        result[f"pass@{k}"] = sum(vals) / len(vals)
    return result


# ---------------------------------------------------------------------------
# vLLM 推理
# ---------------------------------------------------------------------------

def run_eval(model_path: Path, samples: list[dict], n: int,
             temperature: float, tp: int, gpu_mem: float) -> dict:
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n[eval] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=8192,
        trust_remote_code=True,
    )

    # 用 tokenizer chat template 构建 prompt（不传 enable_thinking，与训练一致）
    prompts = [
        tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=True
        )
        for s in samples
    ]

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=0.95,
        max_tokens=8192,
    )

    print(f"[eval] 推理 {len(prompts)} 道题，n={n}, temperature={temperature} ...")
    outputs = llm.generate(prompts, sampling_params)

    per_problem = []
    for i, (output, sample) in enumerate(zip(outputs, samples)):
        generations = [o.text for o in output.outputs]
        correct_flags = [is_correct(extract_boxed(g), sample["ground_truth"]) for g in generations]
        c = sum(correct_flags)
        per_problem.append({
            "idx": i,
            "ground_truth": sample["ground_truth"],
            "n": n,
            "c": c,
            "acc": c / n,
        })

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return per_problem


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="pass@k 批量评测")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--n", type=int, default=8, help="每题采样数（默认8）")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tp", type=int, default=4, help="tensor parallel size（默认4，用满4卡）")
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--output_dir", default="eval_results_pass8")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    k_values = [1, 4, 8] if args.n >= 8 else [1]

    exp_dir = Path(__file__).resolve().parents[1]
    out_dir = exp_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", DATA_ROOT / "train_no_think.parquet"))
    if args.split in ("test", "both"):
        splits.append(("test",  DATA_ROOT / "test_no_think.parquet"))

    models = MODELS
    if args.models:
        models = [(l, p) for l, p in MODELS if l in args.models]

    summary = {}

    for label, model_path in models:
        if not model_path.exists():
            print(f"[SKIP] {label}: 路径不存在 {model_path}")
            continue

        summary[label] = {}

        for split_name, data_file in splits:
            out_file = out_dir / f"{label}_{split_name}.json"
            if out_file.exists():
                print(f"[SKIP] 已有结果: {out_file}")
                with open(out_file) as f:
                    cached = json.load(f)
                summary[label][split_name] = cached["pass_at_k"]
                continue

            print(f"\n{'='*60}")
            print(f"  模型: {label}  数据: {split_name}")
            print(f"{'='*60}")

            samples = load_parquet(data_file)
            per_problem = run_eval(model_path, samples, args.n,
                                   args.temperature, args.tp, args.gpu_mem)
            pass_stats = compute_pass_at_k(per_problem, args.n, k_values)

            print(f"[result] {pass_stats}")

            result = {
                "model": label,
                "model_path": str(model_path),
                "split": split_name,
                "n": args.n,
                "temperature": args.temperature,
                "pass_at_k": pass_stats,
                "per_problem": per_problem,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            summary[label][split_name] = pass_stats

    # 汇总表格
    header = f"  {'模型':<30}  {'split':<6}"
    for k in k_values:
        header += f"  {'pass@'+str(k):>8}"
    print(f"\n{'='*70}")
    print(header)
    print(f"  {'-'*30}  {'-'*6}" + "  --------" * len(k_values))
    for label, splits_res in summary.items():
        for split_name, stats in splits_res.items():
            row = f"  {label:<30}  {split_name:<6}"
            for k in k_values:
                row += f"  {stats.get(f'pass@{k}', 0):>8.4f}"
            print(row)
    print(f"{'='*70}")

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存到: {summary_file}")


if __name__ == "__main__":
    main()
