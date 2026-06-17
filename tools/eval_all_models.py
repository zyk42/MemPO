#!/usr/bin/env python3
"""批量评测所有实验模型在训练集和测试集上的 pass@1 准确率。

模型列表：
  - 原始模型 Qwen3-1.7B（未训练）
  - 5 组实验各自的 global_step_400 和 global_step_480

数据格式（parquet）：
  - prompt: list of {"role": "user", "content": "..."}
  - reward_model: {"ground_truth": "...", "style": "rule"}

用法：
  python tools/eval_all_models.py --split train
  python tools/eval_all_models.py --split test
  python tools/eval_all_models.py --split both
  python tools/eval_all_models.py --split both --tp 4
"""

import argparse
import json
import re
import sys
from pathlib import Path


MOUNT = Path("/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research")
DATA_ROOT = Path.home() / "data/processed/math500"

# 所有待测模型：(标签, 模型路径)
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
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_parquet(data_file: Path) -> list[dict]:
    """加载 parquet，返回 [{"messages": [...], "ground_truth": "..."}]。"""
    import pandas as pd
    df = pd.read_parquet(data_file)
    samples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        # prompt 是 numpy array of dicts
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        gt = row["reward_model"]["ground_truth"]
        samples.append({"messages": messages, "ground_truth": str(gt)})
    return samples


# ---------------------------------------------------------------------------
# 答案提取与验证
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...} 内容。"""
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
# vLLM 推理
# ---------------------------------------------------------------------------

def run_eval(model_path: Path, samples: list[dict], tp: int, gpu_mem: float) -> dict:
    """对一个模型跑 greedy 推理并计算 pass@1。"""
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

    # 用 tokenizer chat template 构建 prompt 字符串
    # Qwen3 /no_think 模式：enable_thinking=False
    prompts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,   # greedy
        max_tokens=8192,
    )

    print(f"[eval] 推理 {len(prompts)} 道题（greedy）...")
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    per_problem = []
    for i, (output, sample) in enumerate(zip(outputs, samples)):
        gen = output.outputs[0].text
        pred = extract_boxed(gen)
        ok = is_correct(pred, sample["ground_truth"])
        if ok:
            correct += 1
        per_problem.append({
            "idx": i,
            "ground_truth": sample["ground_truth"],
            "predicted": pred,
            "correct": ok,
        })

    acc = correct / len(samples)
    print(f"[eval] acc = {correct}/{len(samples)} = {acc:.4f}")

    # 显式释放 vLLM 资源
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return {"acc": acc, "correct": correct, "total": len(samples), "per_problem": per_problem}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="批量评测所有实验模型")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="评测训练集/测试集/两者")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="GPU 内存利用率")
    parser.add_argument("--output_dir", default="eval_results",
                        help="结果保存目录（相对于实验目录）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="只评测指定标签的模型（不指定则全部）")
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parents[1]
    out_dir = exp_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", DATA_ROOT / "train_no_think.parquet"))
    if args.split in ("test", "both"):
        splits.append(("test", DATA_ROOT / "test_no_think.parquet"))

    # 过滤模型列表
    models = MODELS
    if args.models:
        models = [(label, path) for label, path in MODELS if label in args.models]

    # 汇总结果（所有模型 × 所有 split）
    summary = {}   # {label: {split: acc}}

    for label, model_path in models:
        if not model_path.exists():
            print(f"[SKIP] {label}: 路径不存在 {model_path}")
            continue

        summary[label] = {}

        for split_name, data_file in splits:
            print(f"\n{'='*60}")
            print(f"  模型: {label}")
            print(f"  数据: {split_name} ({data_file.name})")
            print(f"{'='*60}")

            out_file = out_dir / f"{label}_{split_name}.json"
            if out_file.exists():
                print(f"[SKIP] 已有结果: {out_file}")
                with open(out_file) as f:
                    cached = json.load(f)
                summary[label][split_name] = cached["acc"]
                continue

            samples = load_parquet(data_file)
            result = run_eval(model_path, samples, args.tp, args.gpu_mem)

            result["model"] = label
            result["model_path"] = str(model_path)
            result["split"] = split_name
            result["data_file"] = str(data_file)

            with open(out_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[save] {out_file}")

            summary[label][split_name] = result["acc"]

    # 打印汇总表格
    print(f"\n{'='*70}")
    print(f"  {'模型':<30}  {'train acc':>10}  {'test acc':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    for label, accs in summary.items():
        train_acc = f"{accs.get('train', 0):.4f}" if "train" in accs else "   -  "
        test_acc  = f"{accs.get('test',  0):.4f}" if "test"  in accs else "   -  "
        print(f"  {label:<30}  {train_acc:>10}  {test_acc:>10}")
    print(f"{'='*70}")

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存到: {summary_file}")


if __name__ == "__main__":
    main()
