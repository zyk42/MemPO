#!/usr/bin/env python3
"""用 vLLM 对模型做 pass@k 评测（数学推理任务）。

pass@k 使用 Codex 论文的无偏估计量：
  pass@k = 1 - C(n-c, k) / C(n, k)
其中 n = 每题采样总数，c = 答对数量，k = 评测值。

支持的数据格式（parquet）：
  必须包含 "problem"（或 "question"）列、"answer"（或 "solution"）列
  可选包含 "extra_info" 列（含 "answer" key）

用法示例：
  # 基本评测（pass@1, pass@4, pass@8）
  python tools/passk_eval.py \\
      --model_dir merged_step240/hf \\
      --data_file ~/data/processed/math500/test.parquet \\
      --n 8 --k 1 4 8

  # 指定更多采样数
  python tools/passk_eval.py \\
      --model_dir merged_step240/hf \\
      --data_file ~/data/processed/math500/test.parquet \\
      --n 16 --k 1 4 8 16 \\
      --temperature 0.7 \\
      --output results_step240.json

  # 对多个 checkpoint 批量评测
  for step in 40 80 120 160 200 240; do
    python tools/passk_eval.py \\
        --model_dir merged/global_step_${step}/hf \\
        --data_file ~/data/processed/math500/test.parquet \\
        --n 8 --k 1 4 8 \\
        --output results/step${step}.json
  done
"""

import argparse
import json
import math
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 答案提取与验证
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """从 \\boxed{...} 中提取最后一个答案。"""
    matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: str) -> str:
    """简单规范化：去空格、统一斜线、去 $ 符号。"""
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("$", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("{", "").replace("}", "")
    return ans


def is_correct(predicted: str | None, ground_truth: str) -> bool:
    """判断预测是否正确（基于字符串规范化匹配）。"""
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(str(ground_truth))


# ---------------------------------------------------------------------------
# pass@k 计算
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏 pass@k 估计（Codex 论文公式）。

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k_stats(results: list[dict], k_values: list[int]) -> dict:
    """
    Args:
        results: [{"n": int, "c": int}, ...]  每道题的采样总数和答对数
        k_values: 要计算的 k 值列表

    Returns:
        {"pass@1": 0.85, "pass@4": 0.92, ...}
    """
    stats = {}
    for k in k_values:
        valid = [r for r in results if r["n"] >= k]
        if not valid:
            stats[f"pass@{k}"] = 0.0
            continue
        avg = sum(pass_at_k(r["n"], r["c"], k) for r in valid) / len(valid)
        stats[f"pass@{k}"] = avg
    return stats


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_dataset(data_file: str) -> list[dict]:
    """加载 parquet 或 jsonl 数据集，返回 [{"problem": ..., "answer": ...}] 列表。"""
    path = Path(data_file)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(data_file)
        rows = df.to_dict(orient="records")
    elif path.suffix in (".jsonl", ".json"):
        rows = []
        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    samples = []
    for row in rows:
        # 兼容不同字段名
        problem = row.get("problem") or row.get("question") or row.get("prompt") or ""
        answer = (
            row.get("answer")
            or row.get("solution")
            or (row.get("extra_info", {}) or {}).get("answer")
            or ""
        )
        samples.append({"problem": str(problem), "answer": str(answer)})

    return samples


def build_prompt(problem: str, system_prompt: str | None = None) -> str:
    """构建推理 prompt（chat 格式，带 think 引导）。"""
    user_content = f"{problem} Let's think step by step and output the final answer within \\boxed{{}}."
    if system_prompt:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"user\n{user_content}\nassistant\n"


# ---------------------------------------------------------------------------
# vLLM 推理
# ---------------------------------------------------------------------------

def run_vllm_inference(
    model_dir: str,
    prompts: list[str],
    n: int,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[list[str]]:
    """用 vLLM 批量推理，返回 outputs[i] = 第 i 个 prompt 的 n 个生成结果。"""
    from vllm import LLM, SamplingParams

    print(f"[vLLM] 加载模型: {model_dir}")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=8192,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    print(f"[vLLM] 推理 {len(prompts)} 道题，每题采样 {n} 次 ...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        texts = [o.text for o in output.outputs]
        results.append(texts)
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vLLM pass@k 评测")
    parser.add_argument("--model_dir", required=True, help="HuggingFace 模型目录")
    parser.add_argument("--data_file", required=True, help="测试数据集（parquet 或 jsonl）")
    parser.add_argument("--n", type=int, default=8, help="每题采样总数（用于 pass@k 估计）")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 8],
                        help="要计算的 k 值列表（默认 1 4 8）")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="采样温度（0 = greedy，推荐 0.6~0.8）")
    parser.add_argument("--max_tokens", type=int, default=8192, help="最大生成 token 数")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="GPU 内存利用率")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评测多少道题（调试用）")
    parser.add_argument("--output", default=None, help="结果保存路径（JSON）")
    parser.add_argument("--system_prompt", default=None, help="系统 prompt（可选）")
    parser.add_argument("--save_generations", action="store_true",
                        help="在结果 JSON 中保存每题的所有生成文本")
    args = parser.parse_args()

    # 验证 k 值
    for k in args.k:
        if k > args.n:
            print(f"[WARN] k={k} > n={args.n}，pass@{k} 将基于不足样本估计")

    # 加载数据集
    print(f"[data] 加载: {args.data_file}")
    samples = load_dataset(args.data_file)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"[data] 共 {len(samples)} 道题")

    # 构建 prompt
    prompts = [build_prompt(s["problem"], args.system_prompt) for s in samples]

    # 推理
    all_generations = run_vllm_inference(
        model_dir=args.model_dir,
        prompts=prompts,
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
    )

    # 评分
    print("[eval] 评分中 ...")
    per_problem = []
    for i, (sample, generations) in enumerate(zip(samples, all_generations)):
        gts = sample["answer"]
        correct_flags = []
        for gen in generations:
            pred = extract_boxed(gen)
            correct_flags.append(is_correct(pred, gts))
        c = sum(correct_flags)
        per_problem.append({
            "idx": i,
            "problem": sample["problem"][:120],
            "answer": gts,
            "n": args.n,
            "c": c,
            "acc": c / args.n,
        })
        if args.save_generations:
            per_problem[-1]["generations"] = generations
            per_problem[-1]["correct_flags"] = correct_flags

    # 计算 pass@k
    pass_stats = compute_pass_at_k_stats(per_problem, args.k)

    # 打印结果
    print(f"\n{'='*50}")
    print(f"  模型: {args.model_dir}")
    print(f"  数据: {args.data_file} ({len(samples)} 道题)")
    print(f"  n={args.n}, temperature={args.temperature}")
    print(f"{'='*50}")
    for metric, val in sorted(pass_stats.items()):
        print(f"  {metric}: {val:.4f} ({val:.2%})")
    print(f"{'='*50}\n")

    # 难题/易题分析
    sorted_by_acc = sorted(per_problem, key=lambda x: x["acc"])
    never_correct = [r for r in sorted_by_acc if r["c"] == 0]
    always_correct = [r for r in sorted_by_acc if r["c"] == args.n]
    print(f"  从不答对（c=0）: {len(never_correct)} 道 ({len(never_correct)/len(samples):.1%})")
    print(f"  全部答对（c=n）: {len(always_correct)} 道 ({len(always_correct)/len(samples):.1%})")

    if never_correct:
        print(f"\n  从不答对的题目示例（前3）：")
        for r in never_correct[:3]:
            print(f"    [{r['answer']}] {r['problem'][:80]}...")

    # 保存结果
    if args.output:
        output_data = {
            "model_dir": args.model_dir,
            "data_file": args.data_file,
            "n_samples": len(samples),
            "n": args.n,
            "temperature": args.temperature,
            "pass_at_k": pass_stats,
            "never_correct": len(never_correct),
            "always_correct": len(always_correct),
            "per_problem": per_problem,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
