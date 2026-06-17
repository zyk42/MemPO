#!/usr/bin/env python3
"""4 GPU 并行评测：每张卡独立跑一个模型，最多 4 个模型同时推理。

相比 tp=4 串行，速度约提升 4x。

用法：
  python tools/eval_parallel.py --split test               # 快速看 test acc
  python tools/eval_parallel.py --split both --n 8         # pass@1/4/8
  python tools/eval_parallel.py --split test --n 1         # 最快，greedy pass@1
"""

import argparse
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 模块级 GPU ID 存储（每个 worker 进程独立）
_worker_gpu_id = None


def _worker_init(gpu_queue):
    """worker 进程启动时从 queue 取一个 GPU ID，整个生命周期固定使用。"""
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()


def run_with_gpu(task):
    gpu_id = _worker_gpu_id
    label, model_path, split_name, data_file, n, temp, out_file = task
    print(f"[GPU {gpu_id}] 启动: {label} / {split_name}")
    return worker(label, model_path, split_name, data_file, n, temp, gpu_id, out_file)


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
]


# ---------------------------------------------------------------------------
# 单 GPU worker（在子进程中运行）
# ---------------------------------------------------------------------------

def worker(label: str, model_path: str, split_name: str, data_file: str,
           n: int, temperature: float, gpu_id: int, out_file: str) -> tuple:
    """单个模型单个 split 的完整评测，在独立子进程中跑。"""
    import gc
    import torch
    import pandas as pd
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 数据加载
    df = pd.read_parquet(data_file)
    samples = []
    for _, row in df.iterrows():
        messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
        samples.append({"messages": messages, "ground_truth": str(row["reward_model"]["ground_truth"])})

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=True)
        for s in samples
    ]

    llm = LLM(model=model_path, tensor_parallel_size=1,
              gpu_memory_utilization=0.7, max_model_len=8192, trust_remote_code=True)

    temperature = temperature if n > 1 else 0.0
    sampling_params = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params)

    def extract_boxed(text):
        matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
        return matches[-1].strip() if matches else None

    def normalize(ans):
        ans = ans.strip().lower()
        ans = re.sub(r"\s+", "", ans)
        return ans.replace("$","").replace("\\left","").replace("\\right","").replace("{","").replace("}","")

    def is_correct(pred, gt):
        return pred is not None and normalize(pred) == normalize(str(gt))

    per_problem = []
    for i, (output, sample) in enumerate(zip(outputs, samples)):
        gens = [o.text for o in output.outputs]
        c = sum(is_correct(extract_boxed(g), sample["ground_truth"]) for g in gens)
        per_problem.append({"idx": i, "ground_truth": sample["ground_truth"], "n": n, "c": c, "acc": c / n})

    # pass@k
    def pass_at_k(n, c, k):
        return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

    k_values = [k for k in [1, 4, 8] if k <= n]
    pass_stats = {f"pass@{k}": sum(pass_at_k(r["n"], r["c"], k) for r in per_problem) / len(per_problem)
                  for k in k_values}

    result = {"model": label, "split": split_name, "n": n, "temperature": temperature,
              "pass_at_k": pass_stats, "per_problem": per_problem}

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return label, split_name, pass_stats


# ---------------------------------------------------------------------------
# 主调度：最多 4 个 worker 并行
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output_dir", default="eval_results_parallel")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parents[1]
    out_dir = exp_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", str(DATA_ROOT / "train_no_think.parquet")))
    if args.split in ("test", "both"):
        splits.append(("test",  str(DATA_ROOT / "test_no_think.parquet")))

    models = MODELS
    if args.models:
        models = [(l, p) for l, p in MODELS if l in args.models]

    # 构建任务列表
    tasks = []
    for label, model_path in models:
        if not model_path.exists():
            print(f"[SKIP] {label}: 路径不存在")
            continue
        for split_name, data_file in splits:
            out_file = str(out_dir / f"{label}_{split_name}.json")
            if Path(out_file).exists():
                print(f"[SKIP] 已有结果: {label} {split_name}")
                continue
            tasks.append((label, str(model_path), split_name, data_file,
                          args.n, args.temperature, out_file))

    print(f"共 {len(tasks)} 个任务，最多 {args.num_gpus} 个并行")

    summary = {}

    # 加载已有结果
    for label, model_path in models:
        for split_name, _ in splits:
            out_file = out_dir / f"{label}_{split_name}.json"
            if out_file.exists():
                with open(out_file) as f:
                    cached = json.load(f)
                summary.setdefault(label, {})[split_name] = cached["pass_at_k"]

    # 并行执行：每个 worker 进程通过 initializer 固定绑定一个 GPU
    with multiprocessing.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in range(args.num_gpus):
            gpu_queue.put(gpu_id)

        futures = {}
        with ProcessPoolExecutor(max_workers=args.num_gpus,
                                  initializer=_worker_init,
                                  initargs=(gpu_queue,)) as executor:
            for task in tasks:
                fut = executor.submit(run_with_gpu, task)
                futures[fut] = (task[0], task[2])  # label, split_name

            for fut in as_completed(futures):
                label, split_name = futures[fut]
                try:
                    _, _, pass_stats = fut.result()
                    print(f"完成: {label} / {split_name}  {pass_stats}")
                    summary.setdefault(label, {})[split_name] = pass_stats
                except Exception as e:
                    print(f"失败: {label} / {split_name}  错误: {e}")

    # 打印汇总
    k_values = [k for k in [1, 4, 8] if k <= args.n]
    header = f"  {'模型':<30}  {'split':<6}" + "".join(f"  {'pass@'+str(k):>8}" for k in k_values)
    print(f"\n{'='*70}")
    print(header)
    print(f"  {'-'*30}  {'-'*6}" + "  --------" * len(k_values))
    for label, splits_res in summary.items():
        for sn, stats in sorted(splits_res.items()):
            row = f"  {label:<30}  {sn:<6}" + "".join(f"  {stats.get(f'pass@{k}', 0):>8.4f}" for k in k_values)
            print(row)
    print(f"{'='*70}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存到: {out_dir}/summary.json")


if __name__ == "__main__":
    main()
