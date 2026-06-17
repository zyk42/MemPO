#!/usr/bin/env python3
"""Phase3 vs Baseline 深度对比分析

分析维度：
1. 总体 pass@k 对比
2. 每道题的 acc 变化分布（win / tie / loss）
3. 按 baseline 难度分组的提升分析（easy / medium / hard）
4. Bootstrap 置信区间（pass@1 是否显著高于 baseline）
5. 训练动态（step400 vs step480 趋势）
"""

import json
import math
import random
from pathlib import Path

RESULTS_DIR = Path("/root/verl/experiments/grpo-forgetting-research/eval_results_pass8")


def load(name, split):
    p = RESULTS_DIR / f"{name}_{split}.json"
    with open(p) as f:
        return json.load(f)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate_passk(per_problem, k):
    vals = [pass_at_k(p["n"], p["c"], k) for p in per_problem]
    return sum(vals) / len(vals)


def bootstrap_ci(per_problem_a, per_problem_b, k=1, n_boot=10000, seed=42):
    """Bootstrap paired difference CI: mean(pass@k_A) - mean(pass@k_B)."""
    rng = random.Random(seed)
    n = len(per_problem_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        a = sum(pass_at_k(per_problem_a[i]["n"], per_problem_a[i]["c"], k) for i in idx) / n
        b = sum(pass_at_k(per_problem_b[i]["n"], per_problem_b[i]["c"], k) for i in idx) / n
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, hi


def per_problem_diff(pp_phase3, pp_baseline):
    """Return list of (idx, acc_phase3, acc_baseline, delta)."""
    base_map = {p["idx"]: p for p in pp_baseline}
    result = []
    for p3 in pp_phase3:
        idx = p3["idx"]
        if idx not in base_map:
            continue
        pb = base_map[idx]
        delta = p3["acc"] - pb["acc"]
        result.append({
            "idx": idx,
            "acc_phase3": p3["acc"],
            "acc_baseline": pb["acc"],
            "delta": delta,
            "c_phase3": p3["c"],
            "c_baseline": pb["c"],
        })
    return result


def difficulty_bucket(acc_baseline):
    if acc_baseline <= 0.125:   # 0 or 1 out of 8 correct
        return "hard (acc≤0.125)"
    elif acc_baseline <= 0.5:
        return "medium (0.125<acc≤0.5)"
    elif acc_baseline < 1.0:
        return "easy (0.5<acc<1.0)"
    else:
        return "mastered (acc=1.0)"


def print_section(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def main():
    for split in ["test", "train"]:
        print_section(f"SPLIT: {split.upper()}")

        baseline = load("baseline_step480", split)
        phase3   = load("phase3_g0.0_step480", split)
        original = load("original", split)

        pp_base = baseline["per_problem"]
        pp_p3   = phase3["per_problem"]
        pp_orig = original["per_problem"]

        # ── 1. 总体 pass@k ──────────────────────────────────────────
        print("\n[1] 总体 pass@k 对比 (step480)")
        print(f"{'方案':<25} {'pass@1':>8} {'pass@4':>8} {'pass@8':>8}")
        print("-" * 52)
        for name, pp in [("original", pp_orig), ("baseline", pp_base), ("phase3", pp_p3)]:
            p1 = aggregate_passk(pp, 1)
            p4 = aggregate_passk(pp, 4)
            p8 = aggregate_passk(pp, 8)
            print(f"{name:<25} {p1:>8.4f} {p4:>8.4f} {p8:>8.4f}")

        delta_p1 = aggregate_passk(pp_p3, 1) - aggregate_passk(pp_base, 1)
        delta_p4 = aggregate_passk(pp_p3, 4) - aggregate_passk(pp_base, 4)
        delta_p8 = aggregate_passk(pp_p3, 8) - aggregate_passk(pp_base, 8)
        print(f"{'Δ phase3 vs baseline':<25} {delta_p1:>+8.4f} {delta_p4:>+8.4f} {delta_p8:>+8.4f}")

        # ── 2. Bootstrap CI ─────────────────────────────────────────
        print("\n[2] Bootstrap 95% CI（phase3 - baseline）")
        for k in [1, 4, 8]:
            lo, hi = bootstrap_ci(pp_p3, pp_base, k=k)
            obs = aggregate_passk(pp_p3, k) - aggregate_passk(pp_base, k)
            sig = "★ 显著" if lo > 0 else ("✗ 不显著" if hi < 0 else "~ 边界")
            print(f"  pass@{k}: Δ={obs:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")

        # ── 3. Per-problem win/tie/loss ──────────────────────────────
        diffs = per_problem_diff(pp_p3, pp_base)
        wins  = [d for d in diffs if d["delta"] > 0]
        ties  = [d for d in diffs if d["delta"] == 0]
        losses= [d for d in diffs if d["delta"] < 0]
        n = len(diffs)

        print(f"\n[3] Per-problem win/tie/loss (n={n})")
        print(f"  phase3 更好: {len(wins):>4} ({100*len(wins)/n:.1f}%)")
        print(f"  持平:        {len(ties):>4} ({100*len(ties)/n:.1f}%)")
        print(f"  baseline 更好: {len(losses):>3} ({100*len(losses)/n:.1f}%)")

        # ── 4. 按难度分组 ────────────────────────────────────────────
        print("\n[4] 按 baseline 难度分组的 Δacc（phase3 - baseline）")
        buckets = {}
        for d in diffs:
            bkt = difficulty_bucket(d["acc_baseline"])
            buckets.setdefault(bkt, []).append(d["delta"])

        print(f"  {'难度档':<30} {'题数':>6} {'平均Δacc':>10} {'胜':>5} {'平':>5} {'负':>5}")
        print("  " + "-" * 62)
        order = ["hard (acc≤0.125)", "medium (0.125<acc≤0.5)",
                 "easy (0.5<acc<1.0)", "mastered (acc=1.0)"]
        for bkt in order:
            if bkt not in buckets:
                continue
            deltas = buckets[bkt]
            avg = sum(deltas) / len(deltas)
            w = sum(1 for x in deltas if x > 0)
            t = sum(1 for x in deltas if x == 0)
            l = sum(1 for x in deltas if x < 0)
            print(f"  {bkt:<30} {len(deltas):>6} {avg:>+10.4f} {w:>5} {t:>5} {l:>5}")

        # ── 5. 最大受益 / 最大受损题目 ────────────────────────────────
        print("\n[5] 受益最多的 10 道题（phase3 - baseline）")
        top_wins = sorted(diffs, key=lambda x: -x["delta"])[:10]
        for d in top_wins:
            print(f"  idx={d['idx']:>4}  baseline_acc={d['acc_baseline']:.3f}  phase3_acc={d['acc_phase3']:.3f}  Δ={d['delta']:+.3f}")

        print("\n[6] 受损最多的 10 道题（phase3 - baseline）")
        top_loss = sorted(diffs, key=lambda x: x["delta"])[:10]
        for d in top_loss:
            print(f"  idx={d['idx']:>4}  baseline_acc={d['acc_baseline']:.3f}  phase3_acc={d['acc_phase3']:.3f}  Δ={d['delta']:+.3f}")

        # ── 6. step400 vs step480 趋势 ────────────────────────────────
        print("\n[7] 训练动态：step400 → step480 趋势")
        base400 = load("baseline_step400", split)
        p3_400  = load("phase3_g0.0_step400", split)
        print(f"  {'方案':<25} {'step400 p@1':>12} {'step480 p@1':>12} {'趋势':>8}")
        print("  " + "-" * 60)
        for name, pp400, pp480 in [
            ("baseline", base400["per_problem"], pp_base),
            ("phase3",   p3_400["per_problem"],  pp_p3),
        ]:
            p1_400 = aggregate_passk(pp400, 1)
            p1_480 = aggregate_passk(pp480, 1)
            trend = p1_480 - p1_400
            print(f"  {name:<25} {p1_400:>12.4f} {p1_480:>12.4f} {trend:>+8.4f}")


if __name__ == "__main__":
    main()
