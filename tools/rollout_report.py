#!/usr/bin/env python3
"""对 verl rollout 日志目录生成详细分析报告。

每个 JSONL 文件对应一个训练 step，每行包含：
  input, output, gts, score, step, acc

报告内容：
  - 整体训练曲线（每步平均 acc）
  - 每道题的首次 vs 末次对比，退化/提升/稳定分类
  - 退化题详情（轨迹、波动率、线性斜率）
  - 高波动题分析
  - 可选：输出 Markdown 报告文件

用法：
  python tools/rollout_report.py --log_dir rollout_logs_alpha0.9_warmup3
  python tools/rollout_report.py --log_dir rollout_logs_alpha0.9_warmup3 --output report.md
  python tools/rollout_report.py --log_dir logs_a --compare_dir logs_b  # 双目录对比
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_log_dir(log_dir: str) -> tuple[dict, dict]:
    """加载目录下所有 JSONL 文件。

    Returns:
        q_step_acc: {question: {step: avg_acc}}
        q_gts:      {question: ground_truth}
    """
    log_path = Path(log_dir)
    files = sorted(
        [f for f in log_path.glob("*.jsonl")],
        key=lambda f: int(f.stem),
    )
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {log_dir}")

    q_step_acc = defaultdict(dict)
    q_gts = {}

    for fpath in files:
        step = int(fpath.stem)
        step_q = defaultdict(list)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                step_q[d["input"]].append(d["acc"])
                if d["input"] not in q_gts:
                    q_gts[d["input"]] = d["gts"]
        for q, accs in step_q.items():
            q_step_acc[q][step] = sum(accs) / len(accs)

    return dict(q_step_acc), q_gts


# ---------------------------------------------------------------------------
# 统计计算
# ---------------------------------------------------------------------------

def linreg_slope(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    xm = sum(xs) / n
    ym = sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    return num / den if den else 0.0


def build_results(q_step_acc: dict, q_gts: dict) -> list[dict]:
    results = []
    for q, step_accs in q_step_acc.items():
        steps = sorted(step_accs.keys())
        accs = [step_accs[s] for s in steps]
        n = len(accs)
        half = max(n // 2, 1)

        first_acc = accs[0]
        last_acc = accs[-1]
        diff_endpoint = last_acc - first_acc
        first_half_avg = statistics.mean(accs[:half])
        last_half_avg = statistics.mean(accs[half:])
        diff_halfavg = last_half_avg - first_half_avg
        slope = linreg_slope(list(range(n)), accs)
        vol = statistics.stdev(accs) if n > 1 else 0.0
        mean_acc = statistics.mean(accs)
        min_acc = min(accs)

        # 分类：三指标都为负才算"真退化"
        truly_degraded = diff_endpoint < 0 and diff_halfavg < 0 and slope < -0.001
        endpoint_degraded = diff_endpoint < 0

        results.append({
            "question": q,
            "gts": q_gts.get(q, ""),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first_acc": first_acc,
            "last_acc": last_acc,
            "diff_endpoint": diff_endpoint,
            "first_half_avg": first_half_avg,
            "last_half_avg": last_half_avg,
            "diff_halfavg": diff_halfavg,
            "slope": slope,
            "volatility": vol,
            "mean_acc": mean_acc,
            "min_acc": min_acc,
            "n_appearances": n,
            "all_accs": accs,
            "all_steps": steps,
            "truly_degraded": truly_degraded,
            "endpoint_degraded": endpoint_degraded,
        })
    return results


def training_curve(q_step_acc: dict) -> list[tuple[int, float]]:
    """按 step 聚合所有题目的平均 acc。"""
    step_all: dict[int, list] = defaultdict(list)
    for step_accs in q_step_acc.values():
        for step, acc in step_accs.items():
            step_all[step].append(acc)
    return [(s, statistics.mean(accs)) for s, accs in sorted(step_all.items())]


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------

def fmt_traj(accs: list[float], steps: list[int]) -> str:
    pairs = [f"{a:.2f}(s{s})" for a, s in zip(accs, steps)]
    if len(pairs) > 8:
        pairs = pairs[:4] + ["..."] + pairs[-2:]
    return " → ".join(pairs)


def render_report(results: list[dict], log_dir: str, compare_results: list[dict] | None = None) -> str:
    lines = []
    A = lines.append

    total = len(results)
    ep_deg = [r for r in results if r["endpoint_degraded"]]
    true_deg = [r for r in results if r["truly_degraded"]]
    improved = [r for r in results if r["diff_endpoint"] > 0]
    stable = [r for r in results if r["diff_endpoint"] == 0]

    A(f"# Rollout 分析报告")
    A(f"\n**目录：** `{log_dir}`")

    steps_all = sorted({s for r in results for s in r["all_steps"]})
    A(f"**训练步数：** {len(steps_all)}（step {steps_all[0]} → {steps_all[-1]}）")
    A(f"**唯一题目数：** {total}")
    A(f"**每题平均出现次数：** {statistics.mean(r['n_appearances'] for r in results):.1f}")

    A(f"\n## 退化/提升汇总\n")
    A(f"| 类别 | 题目数 | 占比 |")
    A(f"|------|--------|------|")
    A(f"| Endpoint 退化（last < first） | {len(ep_deg)} | {len(ep_deg)/total:.1%} |")
    A(f"| 真正退化（3指标全负） | {len(true_deg)} | {len(true_deg)/total:.1%} |")
    A(f"| 提升（last > first） | {len(improved)} | {len(improved)/total:.1%} |")
    A(f"| 不变 | {len(stable)} | {len(stable)/total:.1%} |")

    # 退化幅度分布
    A(f"\n## Endpoint 退化幅度分布\n")
    A(f"| 幅度 | 题目数 |")
    A(f"|------|--------|")
    buckets = [
        ("<-0.5",   [r for r in ep_deg if r["diff_endpoint"] < -0.5]),
        ("-0.5~-0.25", [r for r in ep_deg if -0.5 <= r["diff_endpoint"] < -0.25]),
        ("-0.25~0",  [r for r in ep_deg if -0.25 <= r["diff_endpoint"] < 0]),
    ]
    for label, items in buckets:
        A(f"| {label} | {len(items)} |")

    # 假退化
    false_deg = [r for r in ep_deg if not r["truly_degraded"]]
    A(f"\n> 其中 **假退化**（endpoint 下降但 half-avg 未下降）：**{len(false_deg)}** 道，"
      f"可能是末尾测量噪声，不代表真正遗忘。")

    # 波动率对比
    vol_deg = statistics.mean(r["volatility"] for r in ep_deg) if ep_deg else 0
    vol_stable = statistics.mean(r["volatility"] for r in results if not r["endpoint_degraded"])
    A(f"\n## 波动率分析\n")
    A(f"| 分组 | 平均波动率（acc std across steps） |")
    A(f"|------|----------------------------------|")
    A(f"| Endpoint 退化题 | {vol_deg:.4f} |")
    A(f"| 其他题目 | {vol_stable:.4f} |")
    ratio = vol_deg / vol_stable if vol_stable > 0 else float("inf")
    A(f"\n> 退化题波动率为稳定题的 **{ratio:.1f}x**，说明退化题本身 policy 不稳定。")

    # 真退化题详情
    if true_deg:
        A(f"\n## 真正退化的 {len(true_deg)} 道题（3指标全负）\n")
        A(f"| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | slope | 波动率 |")
        A(f"|-------------|-----|--------|--------|------|-------|--------|")
        for r in sorted(true_deg, key=lambda x: x["slope"]):
            q_short = r["question"][5:65].replace("|", "\\|").replace("\n", " ")
            A(f"| {q_short}... | `{str(r['gts'])[:20]}` | "
              f"{r['first_acc']:.3f} | {r['last_acc']:.3f} | "
              f"{r['diff_endpoint']:+.3f} | {r['slope']:+.4f} | {r['volatility']:.3f} |")

        A(f"\n### 真退化题轨迹\n")
        for r in sorted(true_deg, key=lambda x: x["slope"]):
            q_short = r["question"][5:80].replace("\n", " ")
            A(f"**Q:** {q_short}")
            A(f"```")
            A(fmt_traj(r["all_accs"], r["all_steps"]))
            A(f"```")
            A("")

    # Endpoint 退化题完整列表
    if ep_deg:
        A(f"\n## Endpoint 退化题完整列表（{len(ep_deg)} 道）\n")
        A(f"| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | half-avg差 | 真退化 |")
        A(f"|-------------|-----|--------|--------|------|-----------|--------|")
        for r in sorted(ep_deg, key=lambda x: x["diff_endpoint"]):
            q_short = r["question"][5:60].replace("|", "\\|").replace("\n", " ")
            mark = "✓" if r["truly_degraded"] else ""
            A(f"| {q_short}... | `{str(r['gts'])[:15]}` | "
              f"{r['first_acc']:.3f} | {r['last_acc']:.3f} | "
              f"{r['diff_endpoint']:+.3f} | {r['diff_halfavg']:+.3f} | {mark} |")

    # 高波动题（可能改进空间最大）
    high_vol = sorted(results, key=lambda x: -x["volatility"])[:10]
    A(f"\n## 高波动 Top-10 题目（policy 最不稳定）\n")
    A(f"| 题目（截断） | mean_acc | 波动率 | diff |")
    A(f"|-------------|---------|--------|------|")
    for r in high_vol:
        q_short = r["question"][5:60].replace("|", "\\|").replace("\n", " ")
        A(f"| {q_short}... | {r['mean_acc']:.3f} | {r['volatility']:.3f} | {r['diff_endpoint']:+.3f} |")

    # 对比模式
    if compare_results:
        q2r2 = {r["question"]: r for r in compare_results}
        overlap_deg = [r for r in true_deg if r["question"] in q2r2 and q2r2[r["question"]]["truly_degraded"]]
        A(f"\n## 与对比目录的退化重叠\n")
        A(f"- 当前目录真退化题：{len(true_deg)}")
        A(f"- 对比目录真退化题：{sum(1 for r in compare_results if r['truly_degraded'])}")
        A(f"- 两组均退化：**{len(overlap_deg)}** 道（可能是固有困难题）\n")
        if overlap_deg:
            for r in overlap_deg:
                r2 = q2r2[r["question"]]
                q_short = r["question"][5:80].replace("\n", " ")
                A(f"  - `{q_short[:70]}...`")
                A(f"    当前: {r['first_acc']:.3f}→{r['last_acc']:.3f}  对比: {r2['first_acc']:.3f}→{r2['last_acc']:.3f}")

    return "\n".join(lines)


def print_summary(results: list[dict], log_dir: str):
    """终端简短摘要。"""
    total = len(results)
    ep_deg = sum(1 for r in results if r["endpoint_degraded"])
    true_deg = sum(1 for r in results if r["truly_degraded"])
    improved = sum(1 for r in results if r["diff_endpoint"] > 0)
    steps = sorted({s for r in results for s in r["all_steps"]})

    print(f"\n{'='*60}")
    print(f"  目录: {log_dir}")
    print(f"  步数范围: step {steps[0]} → {steps[-1]} (共 {len(steps)} 步)")
    print(f"  唯一题目: {total}")
    print(f"  Endpoint 退化: {ep_deg} ({ep_deg/total:.1%})")
    print(f"  真正退化 (3指标): {true_deg} ({true_deg/total:.1%})")
    print(f"  提升: {improved} ({improved/total:.1%})")
    print(f"{'='*60}\n")

    print("退化最严重的 5 道题：")
    for r in sorted(results, key=lambda x: x["diff_endpoint"])[:5]:
        q_short = r["question"][5:75].replace("\n", " ")
        print(f"  diff={r['diff_endpoint']:+.3f} slope={r['slope']:+.4f} "
              f"vol={r['volatility']:.3f}  gts={str(r['gts'])[:15]}")
        print(f"  {q_short}...")
    print()


# ---------------------------------------------------------------------------
# 训练曲线打印
# ---------------------------------------------------------------------------

def print_training_curve(curve: list[tuple[int, float]], width: int = 50):
    print("训练曲线（每 step 平均 acc）：")
    if not curve:
        return
    max_acc = max(a for _, a in curve)
    min_acc = min(a for _, a in curve)
    rng = max_acc - min_acc or 0.01

    # 每隔 N 步打印一行
    stride = max(1, len(curve) // 30)
    for i, (step, acc) in enumerate(curve):
        if i % stride != 0 and i != len(curve) - 1:
            continue
        bar_len = int((acc - min_acc) / rng * width)
        bar = "█" * bar_len
        print(f"  step {step:4d} | {bar:<{width}} | {acc:.4f}")
    print()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="生成 rollout 日志详细分析报告")
    parser.add_argument("--log_dir", required=True, help="rollout 日志目录（含 *.jsonl）")
    parser.add_argument("--compare_dir", default=None, help="对比目录（可选，用于双组对比）")
    parser.add_argument("--output", default=None, help="Markdown 报告输出路径（不指定则仅打印摘要）")
    parser.add_argument("--no_curve", action="store_true", help="不打印训练曲线")
    args = parser.parse_args()

    print(f"加载 {args.log_dir} ...")
    q_step_acc, q_gts = load_log_dir(args.log_dir)
    results = build_results(q_step_acc, q_gts)

    compare_results = None
    if args.compare_dir:
        print(f"加载对比目录 {args.compare_dir} ...")
        q2, g2 = load_log_dir(args.compare_dir)
        compare_results = build_results(q2, g2)

    # 终端摘要
    print_summary(results, args.log_dir)

    if not args.no_curve:
        curve = training_curve(q_step_acc)
        print_training_curve(curve)

    # Markdown 报告
    if args.output:
        report = render_report(results, args.log_dir, compare_results)
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已保存到: {out_path}")
    else:
        # 不保存文件时，打印详细退化列表
        ep_deg = [r for r in results if r["endpoint_degraded"]]
        if ep_deg:
            print(f"Endpoint 退化题（{len(ep_deg)} 道）：")
            for r in sorted(ep_deg, key=lambda x: x["diff_endpoint"]):
                truly = "[真退化]" if r["truly_degraded"] else "[假退化]"
                q_short = r["question"][5:80].replace("\n", " ")
                print(f"  {truly} diff={r['diff_endpoint']:+.3f} halfavg={r['diff_halfavg']:+.3f} "
                      f"slope={r['slope']:+.4f} gts={str(r['gts'])[:15]}")
                print(f"    轨迹: {fmt_traj(r['all_accs'], r['all_steps'])}")


if __name__ == "__main__":
    main()
