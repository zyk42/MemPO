#!/usr/bin/env python3
"""对比多组实验的 rollout 日志，输出退化对比表格和训练曲线。

适合对比：baseline vs EMA-GRPO vs EMA+filter vs EMA+filter+KL 等多组实验。

用法：
  python tools/compare_experiments.py \\
      --dirs rollout_logs_baseline rollout_logs_alpha0.9_warmup3 \\
      --names Baseline EMA-GRPO \\
      --output comparison.md

  # 对比所有以 rollout_logs_ 开头的目录
  python tools/compare_experiments.py \\
      --dirs rollout_logs_* \\
      --output comparison.md
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 复用 rollout_report.py 的加载和计算逻辑
# ---------------------------------------------------------------------------

def load_log_dir(log_dir: str):
    log_path = Path(log_dir)
    files = sorted(log_path.glob("*.jsonl"), key=lambda f: int(f.stem))
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


def linreg_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    xm, ym = sum(xs) / n, sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    return num / den if den else 0.0


def build_results(q_step_acc, q_gts):
    results = {}
    for q, step_accs in q_step_acc.items():
        steps = sorted(step_accs.keys())
        accs = [step_accs[s] for s in steps]
        n = len(accs)
        half = max(n // 2, 1)
        diff_ep = accs[-1] - accs[0]
        diff_half = statistics.mean(accs[half:]) - statistics.mean(accs[:half])
        slope = linreg_slope(list(range(n)), accs)
        vol = statistics.stdev(accs) if n > 1 else 0.0
        results[q] = {
            "gts": q_gts.get(q, ""),
            "first_acc": accs[0],
            "last_acc": accs[-1],
            "diff_ep": diff_ep,
            "diff_half": diff_half,
            "slope": slope,
            "vol": vol,
            "mean_acc": statistics.mean(accs),
            "accs": accs,
            "steps": steps,
            "truly_degraded": diff_ep < 0 and diff_half < 0 and slope < -0.001,
            "ep_degraded": diff_ep < 0,
        }
    return results


def training_curve(q_step_acc):
    step_all = defaultdict(list)
    for step_accs in q_step_acc.values():
        for step, acc in step_accs.items():
            step_all[step].append(acc)
    return sorted((s, statistics.mean(accs)) for s, accs in step_all.items())


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def render_comparison(exps: list[dict]) -> str:
    """
    exps: [{"name": str, "dir": str, "results": dict, "curve": list}, ...]
    """
    lines = []
    A = lines.append

    A("# 多实验对比报告\n")

    # 实验列表
    A("## 实验组\n")
    A("| # | 名称 | 目录 | 题目数 | 步数 |")
    A("|---|------|------|--------|------|")
    for i, exp in enumerate(exps):
        steps = sorted({s for r in exp["results"].values() for s in r["steps"]})
        A(f"| {i+1} | **{exp['name']}** | `{exp['dir']}` | "
          f"{len(exp['results'])} | {steps[0]}→{steps[-1]} ({len(steps)}步) |")

    # 核心对比表
    A("\n## 退化统计对比\n")
    A("| 指标 | " + " | ".join(f"**{e['name']}**" for e in exps) + " |")
    A("|------|" + "|".join(["---"] * len(exps)) + "|")

    def row(label, fn):
        vals = [fn(e["results"]) for e in exps]
        return f"| {label} | " + " | ".join(str(v) for v in vals) + " |"

    total = len(exps[0]["results"])  # 以第一组为基准
    A(row("Endpoint 退化题数", lambda r: sum(1 for x in r.values() if x["ep_degraded"])))
    A(row("真正退化题数（3指标）", lambda r: sum(1 for x in r.values() if x["truly_degraded"])))
    A(row("提升题数", lambda r: sum(1 for x in r.values() if x["diff_ep"] > 0)))
    A(row("平均 acc（所有题）", lambda r: f"{statistics.mean(x['mean_acc'] for x in r.values()):.4f}"))
    A(row("平均波动率", lambda r: f"{statistics.mean(x['vol'] for x in r.values()):.4f}"))
    A(row("退化题平均波动率",
          lambda r: f"{statistics.mean((x['vol'] for x in r.values() if x['ep_degraded']), default=0):.4f}"))

    # 训练曲线对比（最后一步 acc）
    A("\n## 末尾训练 acc 对比\n")
    A("| 实验 | 末尾 step | 末尾平均 acc | 最高平均 acc |")
    A("|------|----------|-------------|-------------|")
    for exp in exps:
        curve = exp["curve"]
        last_step, last_acc = curve[-1]
        max_acc = max(a for _, a in curve)
        A(f"| **{exp['name']}** | {last_step} | {last_acc:.4f} | {max_acc:.4f} |")

    # 训练曲线文字图（ASCII）
    A("\n## 训练曲线（每步平均 acc）\n")
    A("```")
    # 对齐所有实验的 steps
    all_steps = sorted({s for e in exps for s, _ in e["curve"]})
    exp_curve_map = []
    for exp in exps:
        d = dict(exp["curve"])
        exp_curve_map.append(d)

    stride = max(1, len(all_steps) // 25)
    header = f"{'step':>6} | " + " | ".join(f"{e['name']:>12}" for e in exps)
    A(header)
    A("-" * len(header))
    for i, step in enumerate(all_steps):
        if i % stride != 0 and i != len(all_steps) - 1:
            continue
        vals = []
        for d in exp_curve_map:
            acc = d.get(step)
            vals.append(f"{acc:>12.4f}" if acc is not None else f"{'N/A':>12}")
        A(f"{step:>6} | " + " | ".join(vals))
    A("```")

    # 两两重叠分析（仅适用于 2 组）
    if len(exps) == 2:
        r1 = exps[0]["results"]
        r2 = exps[1]["results"]
        common_qs = set(r1.keys()) & set(r2.keys())
        both_deg = [q for q in common_qs if r1[q]["truly_degraded"] and r2[q]["truly_degraded"]]
        only1 = [q for q in common_qs if r1[q]["truly_degraded"] and not r2[q]["truly_degraded"]]
        only2 = [q for q in common_qs if not r1[q]["truly_degraded"] and r2[q]["truly_degraded"]]

        A(f"\n## 退化题重叠分析（{exps[0]['name']} vs {exps[1]['name']}）\n")
        A(f"| 分类 | 题目数 | 说明 |")
        A(f"|------|--------|------|")
        A(f"| 两组均退化 | {len(both_deg)} | 可能是固有困难题，方法无关 |")
        A(f"| 仅 {exps[0]['name']} 退化 | {len(only1)} | {exps[1]['name']} 成功保护 |")
        A(f"| 仅 {exps[1]['name']} 退化 | {len(only2)} | {exps[0]['name']} 成功保护 |")

        if both_deg:
            A(f"\n### 两组均退化的固有困难题\n")
            for q in sorted(both_deg, key=lambda q: r1[q]["diff_ep"]):
                q_short = q[5:80].replace("\n", " ")
                A(f"- **GTS=`{str(r1[q]['gts'])[:20]}`**: {q_short}...")
                A(f"  - {exps[0]['name']}: {r1[q]['first_acc']:.3f}→{r1[q]['last_acc']:.3f} "
                  f"(slope={r1[q]['slope']:+.4f})")
                A(f"  - {exps[1]['name']}: {r2[q]['first_acc']:.3f}→{r2[q]['last_acc']:.3f} "
                  f"(slope={r2[q]['slope']:+.4f})")

        if only1:
            A(f"\n### 仅 {exps[0]['name']} 退化（{exps[1]['name']} 成功保护）\n")
            for q in only1[:10]:
                q_short = q[5:80].replace("\n", " ")
                A(f"- {q_short}...")
                A(f"  - {exps[0]['name']}: {r1[q]['first_acc']:.3f}→{r1[q]['last_acc']:.3f}")
                A(f"  - {exps[1]['name']}: {r2[q]['first_acc']:.3f}→{r2[q]['last_acc']:.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="对比多组 rollout 日志实验")
    parser.add_argument("--dirs", nargs="+", required=True, help="rollout 日志目录列表")
    parser.add_argument("--names", nargs="*", default=None,
                        help="实验名称（顺序与 --dirs 对应，不指定则用目录名）")
    parser.add_argument("--output", default=None, help="Markdown 报告输出路径")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.dirs):
        parser.error("--names 数量必须与 --dirs 数量一致")

    names = args.names or [Path(d).name for d in args.dirs]

    exps = []
    for name, d in zip(names, args.dirs):
        print(f"加载 [{name}]: {d} ...")
        try:
            q_step_acc, q_gts = load_log_dir(d)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}，跳过")
            continue
        results = build_results(q_step_acc, q_gts)
        curve = training_curve(q_step_acc)
        exps.append({"name": name, "dir": d, "results": results, "curve": curve})

    if len(exps) < 1:
        print("没有可用的实验目录，退出")
        return

    # 终端摘要
    print(f"\n{'='*70}")
    print(f"{'实验':<30} {'Endpoint退化':>12} {'真退化':>8} {'提升':>8} {'末尾acc':>10}")
    print(f"{'='*70}")
    for exp in exps:
        r = exp["results"]
        ep_deg = sum(1 for x in r.values() if x["ep_degraded"])
        true_deg = sum(1 for x in r.values() if x["truly_degraded"])
        imp = sum(1 for x in r.values() if x["diff_ep"] > 0)
        last_acc = exp["curve"][-1][1] if exp["curve"] else 0
        print(f"{exp['name']:<30} {ep_deg:>12} {true_deg:>8} {imp:>8} {last_acc:>10.4f}")
    print(f"{'='*70}\n")

    if args.output:
        report = render_comparison(exps)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已保存到: {out_path}")


if __name__ == "__main__":
    main()
