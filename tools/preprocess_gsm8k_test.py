#!/usr/bin/env python3
"""将 GSM8K 测试集（1319题）按 seed=42 打乱后分成训练集（1000题）和测试集（319题），
输出格式与 math500 no_think 版本一致（data_source="gsm8k"，\\boxed{} 输出）。

用法：
  python3 tools/preprocess_gsm8k_test.py
  python3 tools/preprocess_gsm8k_test.py --train_size 1000 --output_dir ~/data/processed/gsm8k_test_split
"""
import argparse
import json
import os
import random
import re

import pandas as pd


def extract_gt(answer_str: str) -> str:
    """从 GSM8K answer 字段中提取 #### 后的数字答案。"""
    m = re.search(r"####\s*([\-0-9,\.]+)", answer_str)
    assert m, f"No #### found in: {answer_str!r}"
    return m.group(1).replace(",", "").strip()


def make_row(example: dict, idx: int, split: str) -> dict:
    question = example["question"].strip()
    gt = extract_gt(example["answer"])
    instruction = (
        "/no_think " + question +
        " Let's think step by step and output the final answer within \\boxed{}."
    )
    return {
        "data_source": "gsm8k",
        "prompt": [{"role": "user", "content": instruction}],
        "ability": "math",
        "reward_model": {"ground_truth": gt, "style": "rule"},
        "extra_info": {"split": split, "index": idx,
                       "answer": example["answer"], "question": question},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.expanduser("~/data/gsm8k/test.jsonl"))
    parser.add_argument("--output_dir", default=os.path.expanduser("~/data/processed/gsm8k_test_split"))
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        examples = [json.loads(l) for l in f]

    print(f"Loaded {len(examples)} examples from {args.input}")

    rng = random.Random(args.seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    train_idx = indices[:args.train_size]
    test_idx = indices[args.train_size:]

    train_rows = [make_row(examples[i], j, "train") for j, i in enumerate(train_idx)]
    test_rows  = [make_row(examples[i], j, "test")  for j, i in enumerate(test_idx)]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_no_think.parquet")
    test_path  = os.path.join(args.output_dir, "test_no_think.parquet")

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(test_rows).to_parquet(test_path, index=False)

    print(f"Train: {len(train_rows)} rows → {train_path}")
    print(f"Test:  {len(test_rows)} rows → {test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
