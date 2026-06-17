# Rollout 分析报告

**目录：** `experiments/grpo-forgetting-research/rollout_logs_phase2_alpha0.9_gamma2.0`
**训练步数：** 480（step 1 → 480）
**唯一题目数：** 398
**每题平均出现次数：** 19.3

## 退化/提升汇总

| 类别 | 题目数 | 占比 |
|------|--------|------|
| Endpoint 退化（last < first） | 40 | 10.1% |
| 真正退化（3指标全负） | 8 | 2.0% |
| 提升（last > first） | 120 | 30.2% |
| 不变 | 238 | 59.8% |

## Endpoint 退化幅度分布

| 幅度 | 题目数 |
|------|--------|
| <-0.5 | 0 |
| -0.5~-0.25 | 3 |
| -0.25~0 | 37 |

> 其中 **假退化**（endpoint 下降但 half-avg 未下降）：**32** 道，可能是末尾测量噪声，不代表真正遗忘。

## 波动率分析

| 分组 | 平均波动率（acc std across steps） |
|------|----------------------------------|
| Endpoint 退化题 | 0.1327 |
| 其他题目 | 0.0762 |

> 退化题波动率为稳定题的 **1.7x**，说明退化题本身 policy 不稳定。

## 真正退化的 8 道题（3指标全负）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | slope | 波动率 |
|-------------|-----|--------|--------|------|-------|--------|
| /no_think Linda, Sherry, June, and Connie walked around thei... | `59` | 0.125 | 0.000 | -0.125 | -0.0140 | 0.155 |
| /no_think What is the range of the function $y=\log_2 (\sqrt... | `(-\infty, 0]` | 0.250 | 0.125 | -0.125 | -0.0108 | 0.133 |
| /no_think A group of $N$ students, where $N < 50$, is on a f... | `66` | 1.000 | 0.875 | -0.125 | -0.0048 | 0.046 |
| /no_think Let $A = (1,-11,2),$ $B = (3,-4,1),$ and $C = (-2,... | `120^\circ` | 1.000 | 0.875 | -0.125 | -0.0036 | 0.040 |
| /no_think A curve is parameterized by \[(x,y) = (t^3 + 7, -3... | `(15,-29)` | 1.000 | 0.750 | -0.250 | -0.0036 | 0.056 |
| /no_think The parallelogram bounded by the lines $y=ax+c$, $... | `16` | 0.125 | 0.000 | -0.125 | -0.0036 | 0.051 |
| /no_think Twelve friends met for dinner at Oscar's Overstuff... | `8` | 0.250 | 0.125 | -0.125 | -0.0035 | 0.073 |
| /no_think The 7th and 8th grades have enrollments of 520 and... | `10` | 1.000 | 0.750 | -0.250 | -0.0031 | 0.076 |

### 真退化题轨迹

**Q:** /no_think Linda, Sherry, June, and Connie walked around their neighborhoods
```
0.12(s31) → 0.12(s69) → 0.25(s78) → 0.38(s104) → ... → 0.00(s437) → 0.00(s464)
```

**Q:** /no_think What is the range of the function $y=\log_2 (\sqrt{\sin x})$ for 
```
0.25(s21) → 0.25(s39) → 0.50(s55) → 0.38(s74) → ... → 0.12(s441) → 0.12(s459)
```

**Q:** /no_think A group of $N$ students, where $N < 50$, is on a field trip. If t
```
1.00(s2) → 1.00(s40) → 1.00(s66) → 1.00(s74) → ... → 0.88(s455) → 0.88(s465)
```

**Q:** /no_think Let $A = (1,-11,2),$ $B = (3,-4,1),$ and $C = (-2,1,-1).$  Comput
```
1.00(s11) → 1.00(s51) → 1.00(s74) → 1.00(s99) → ... → 1.00(s452) → 0.88(s457)
```

**Q:** /no_think A curve is parameterized by \[(x,y) = (t^3 + 7, -3t^2 - 6t - 5).\
```
1.00(s2) → 1.00(s29) → 1.00(s66) → 1.00(s91) → ... → 1.00(s440) → 0.75(s470)
```

**Q:** /no_think The parallelogram bounded by the lines $y=ax+c$, $y=ax+d$, $y=bx+
```
0.12(s4) → 0.12(s25) → 0.00(s60) → 0.12(s80) → ... → 0.00(s438) → 0.00(s475)
```

**Q:** /no_think Twelve friends met for dinner at Oscar's Overstuffed Oyster House
```
0.25(s15) → 0.00(s38) → 0.12(s63) → 0.00(s74) → ... → 0.00(s449) → 0.12(s466)
```

**Q:** /no_think The 7th and 8th grades have enrollments of 520 and 650 respective
```
1.00(s5) → 0.88(s40) → 1.00(s58) → 1.00(s96) → ... → 0.88(s436) → 0.75(s459)
```


## Endpoint 退化题完整列表（40 道）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | half-avg差 | 真退化 |
|-------------|-----|--------|--------|------|-----------|--------|
| /no_think The area of $\triangle ABC$ is 6 square centi... | `54` | 0.500 | 0.000 | -0.500 | +0.062 |  |
| /no_think What is the domain of the function $f(x) = \f... | `(2,12) \cup (12` | 1.000 | 0.500 | -0.500 | +0.087 |  |
| /no_think If \[f(n + 1) = (-1)^{n + 1} n - 2f(n)\]for $... | `331` | 0.500 | 0.125 | -0.375 | +0.176 |  |
| /no_think A curve is parameterized by \[(x,y) = (t^3 + ... | `(15,-29)` | 1.000 | 0.750 | -0.250 | -0.025 | ✓ |
| /no_think A $90^\circ$ rotation around $-4 - 5i$ in the... | `1 - 12i` | 1.000 | 0.750 | -0.250 | +0.075 |  |
| /no_think In circle $J$, $HO$ and $HN$ are tangent to t... | `180^\circ` | 0.750 | 0.500 | -0.250 | +0.100 |  |
| /no_think The 7th and 8th grades have enrollments of 52... | `10` | 1.000 | 0.750 | -0.250 | -0.012 | ✓ |
| /no_think Find the value of $\log_2{3} \cdot \log_3{4} ... | `3` | 0.750 | 0.500 | -0.250 | +0.022 |  |
| /no_think The set of points $(x,y,z)$ that satisfy \[2x... | `90^\circ` | 0.500 | 0.250 | -0.250 | +0.261 |  |
| /no_think For $0 \le x \le 40$ and $0 \le y \le 50,$ fi... | `70 \sqrt{2}` | 0.250 | 0.000 | -0.250 | +0.138 |  |
| /no_think Charlie is riding on his unicycle. If the uni... | `12\pi` | 1.000 | 0.750 | -0.250 | -0.001 |  |
| /no_think Let $f$ be the function defined by $f(x) = x^... | `34` | 1.000 | 0.750 | -0.250 | +0.073 |  |
| /no_think What is the sum of the digits in the terminat... | `14` | 0.750 | 0.625 | -0.125 | +0.200 |  |
| /no_think Six witches and ten sorcerers are at an arcan... | `60` | 0.750 | 0.625 | -0.125 | +0.129 |  |
| /no_think A group of $N$ students, where $N < 50$, is o... | `66` | 1.000 | 0.875 | -0.125 | -0.037 | ✓ |
| /no_think The parallelogram bounded by the lines $y=ax+... | `16` | 0.125 | 0.000 | -0.125 | -0.025 | ✓ |
| /no_think Let $f$ be defined by  \[f(x) = \left\{ \begi... | `0` | 0.875 | 0.750 | -0.125 | +0.137 |  |
| /no_think Find the units digit of $18^6.$ Let's think s... | `4` | 1.000 | 0.875 | -0.125 | +0.113 |  |
| /no_think Recently, Frank took a one-hundred question a... | `56` | 0.875 | 0.750 | -0.125 | -0.050 |  |
| /no_think Tom got a Mr. Potato Head for his birthday. I... | `64` | 0.125 | 0.000 | -0.125 | +0.050 |  |
| /no_think Let $f(x)=\left\lfloor\left(-\frac58\right)^x... | `3` | 0.625 | 0.500 | -0.125 | +0.086 |  |
| /no_think Each week, between 30 and 50 students show up... | `41` | 0.500 | 0.375 | -0.125 | +0.132 |  |
| /no_think Suppose that I have $6$ different books, $2$ ... | `480` | 1.000 | 0.875 | -0.125 | +0.001 |  |
| /no_think One line is defined by \[\begin{pmatrix} 3 \\... | `7` | 0.250 | 0.125 | -0.125 | +0.000 |  |
| /no_think The medians $AD$, $BE$, and $CF$ of triangle ... | `8` | 0.375 | 0.250 | -0.125 | +0.175 |  |
| /no_think A regular pentagon is rotated counterclockwis... | `72` | 0.875 | 0.750 | -0.125 | +0.062 |  |
| /no_think Let $A = (1,-11,2),$ $B = (3,-4,1),$ and $C =... | `120^\circ` | 1.000 | 0.875 | -0.125 | -0.028 | ✓ |
| /no_think Find the degree measure of the least positive... | `120^\circ` | 0.875 | 0.750 | -0.125 | +0.111 |  |
| /no_think Compute \[\frac{1}{\cos^2 10^\circ} + \frac{1... | `12` | 0.625 | 0.500 | -0.125 | +0.050 |  |
| /no_think A right cylindrical tank with circular bases ... | `\sqrt{5}` | 1.000 | 0.875 | -0.125 | +0.000 |  |
| /no_think Consider the rectangular region with the foll... | `63` | 1.000 | 0.875 | -0.125 | +0.074 |  |
| /no_think Twelve friends met for dinner at Oscar's Over... | `8` | 0.250 | 0.125 | -0.125 | -0.031 | ✓ |
| /no_think Expand and simplify completely: \begin{align*... | `x^3+3x-6` | 0.625 | 0.500 | -0.125 | +0.275 |  |
| /no_think How many sides would there be in a convex pol... | `8` | 1.000 | 0.875 | -0.125 | -0.014 |  |
| /no_think Compute \[\sum_{n = 1}^\infty \frac{F_{n + 1}... | `2` | 0.625 | 0.500 | -0.125 | +0.050 |  |
| /no_think What is the range of the function $y=\log_2 (... | `(-\infty, 0]` | 0.250 | 0.125 | -0.125 | -0.083 | ✓ |
| /no_think An equilateral triangle is inscribed in the p... | `16 \sqrt{3}` | 1.000 | 0.875 | -0.125 | +0.112 |  |
| /no_think Let $a,$ $b,$ $c,$ $d$ be distinct complex nu... | `0` | 0.125 | 0.000 | -0.125 | +0.037 |  |
| /no_think Let $F(z)=\frac{z+i}{z-i}$ for all complex nu... | `1+274i` | 0.125 | 0.000 | -0.125 | +0.037 |  |
| /no_think Linda, Sherry, June, and Connie walked around... | `59` | 0.125 | 0.000 | -0.125 | -0.201 | ✓ |

## 高波动 Top-10 题目（policy 最不稳定）

| 题目（截断） | mean_acc | 波动率 | diff |
|-------------|---------|--------|------|
| /no_think If $A$ is the sum of the positive divisors of... | 0.651 | 0.360 | +0.500 |
| /no_think What is the smallest number which is one less... | 0.738 | 0.300 | +0.375 |
| /no_think What is the result when the greatest common f... | 0.493 | 0.296 | +1.000 |
| /no_think Expand and simplify completely: \begin{align*... | 0.588 | 0.257 | -0.125 |
| /no_think If each point of the circle $x^2 + y^2 = 25$ ... | 0.762 | 0.256 | +0.250 |
| /no_think The set of points $(x,y,z)$ that satisfy \[2x... | 0.651 | 0.255 | -0.250 |
| /no_think Remmy wants to divide $10$ by $\frac{2}{3}$, ... | 0.794 | 0.254 | +0.375 |
| /no_think Let $\mathbf{A}$ be a matrix such that \[\mat... | 0.738 | 0.253 | +0.125 |
| /no_think If $a$ and $b$ are positive integers such tha... | 0.461 | 0.250 | +0.500 |
| /no_think For how many real values of $x$ is $\sqrt{120... | 0.750 | 0.250 | +0.500 |