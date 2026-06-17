# Rollout 分析报告

**目录：** `rollout_logs_phase4_alpha0.9_gamma0.5`
**训练步数：** 480（step 1 → 480）
**唯一题目数：** 398
**每题平均出现次数：** 19.3

## 退化/提升汇总

| 类别 | 题目数 | 占比 |
|------|--------|------|
| Endpoint 退化（last < first） | 40 | 10.1% |
| 真正退化（3指标全负） | 18 | 4.5% |
| 提升（last > first） | 135 | 33.9% |
| 不变 | 223 | 56.0% |

## Endpoint 退化幅度分布

| 幅度 | 题目数 |
|------|--------|
| <-0.5 | 1 |
| -0.5~-0.25 | 5 |
| -0.25~0 | 34 |

> 其中 **假退化**（endpoint 下降但 half-avg 未下降）：**22** 道，可能是末尾测量噪声，不代表真正遗忘。

## 波动率分析

| 分组 | 平均波动率（acc std across steps） |
|------|----------------------------------|
| Endpoint 退化题 | 0.1226 |
| 其他题目 | 0.0789 |

> 退化题波动率为稳定题的 **1.6x**，说明退化题本身 policy 不稳定。

## 真正退化的 18 道题（3指标全负）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | slope | 波动率 |
|-------------|-----|--------|--------|------|-------|--------|
| /no_think $\overline{BC}$ is parallel to the segment through... | `28` | 0.500 | 0.375 | -0.125 | -0.0147 | 0.183 |
| /no_think If $re^{i \theta}$ is a root of \[z^8 - z^7 + z^6 ... | `8 \pi` | 0.625 | 0.000 | -0.625 | -0.0140 | 0.155 |
| /no_think What is the range of the function $y=\log_2 (\sqrt... | `(-\infty, 0]` | 0.500 | 0.000 | -0.500 | -0.0115 | 0.191 |
| /no_think In the circle with center $Q$, radii $AQ$ and $BQ$... | `\frac{14}{3}` | 0.375 | 0.000 | -0.375 | -0.0114 | 0.159 |
| /no_think Two-thirds of the students at Baker Middle School ... | `1251` | 1.000 | 0.625 | -0.375 | -0.0105 | 0.134 |
| /no_think Jim and Martha are standing together at the corner... | `200` | 0.750 | 0.375 | -0.375 | -0.0104 | 0.227 |
| /no_think If $x^3$ is a positive factor of $10!,$ how many p... | `6` | 1.000 | 0.750 | -0.250 | -0.0058 | 0.115 |
| /no_think Two numbers, $x$ and $y$ are selected at random fr... | `\frac{1}{2}` | 0.125 | 0.000 | -0.125 | -0.0057 | 0.075 |
| /no_think If \[f(n + 1) = (-1)^{n + 1} n - 2f(n)\]for $n \ge... | `331` | 0.500 | 0.250 | -0.250 | -0.0050 | 0.174 |
| /no_think The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$ and ... | `40` | 0.250 | 0.125 | -0.125 | -0.0048 | 0.138 |
| /no_think The two-digit number $``B6,''$ where $B$ is the te... | `2` | 0.875 | 0.750 | -0.125 | -0.0047 | 0.212 |
| /no_think Convert the point $(0,3)$ in rectangular coordinat... | `\left( 3, \frac{\pi}` | 1.000 | 0.750 | -0.250 | -0.0044 | 0.059 |
| /no_think Steve says to Jon, "I am thinking of a polynomial ... | `440` | 0.250 | 0.000 | -0.250 | -0.0039 | 0.115 |
| /no_think If $f(x)=5x^2+3x+4$, what is the value of $f(-2)$?... | `18` | 1.000 | 0.875 | -0.125 | -0.0034 | 0.038 |
| /no_think Determine the modulo 4 remainder of the following ... | `2` | 1.000 | 0.875 | -0.125 | -0.0034 | 0.040 |
| /no_think Let $a,$ $b,$ and $c$ be distinct real numbers.  F... | `0` | 0.375 | 0.250 | -0.125 | -0.0031 | 0.125 |
| /no_think Find all $p$ which satisfy both the inequalities $... | `\left(\frac{3}{5},\f` | 1.000 | 0.875 | -0.125 | -0.0020 | 0.029 |
| /no_think In the diagram below, we have $\overline{ST}\paral... | `75^\circ` | 0.125 | 0.000 | -0.125 | -0.0012 | 0.046 |

### 真退化题轨迹

**Q:** /no_think $\overline{BC}$ is parallel to the segment through $A$, and $AB =
```
0.50(s16) → 0.25(s27) → 0.50(s72) → 0.75(s81) → ... → 0.12(s444) → 0.38(s465)
```

**Q:** /no_think If $re^{i \theta}$ is a root of \[z^8 - z^7 + z^6 - z^5 + z^4 - z
```
0.62(s9) → 0.25(s68) → 0.12(s94) → 0.25(s117) → ... → 0.00(s444) → 0.00(s468)
```

**Q:** /no_think What is the range of the function $y=\log_2 (\sqrt{\sin x})$ for 
```
0.50(s21) → 0.12(s39) → 0.25(s55) → 0.25(s74) → ... → 0.62(s441) → 0.00(s459)
```

**Q:** /no_think In the circle with center $Q$, radii $AQ$ and $BQ$ form a right a
```
0.38(s8) → 0.25(s39) → 0.12(s56) → 0.38(s90) → ... → 0.25(s454) → 0.00(s471)
```

**Q:** /no_think Two-thirds of the students at Baker Middle School take music. The
```
1.00(s16) → 1.00(s46) → 1.00(s66) → 1.00(s92) → ... → 0.62(s448) → 0.62(s466)
```

**Q:** /no_think Jim and Martha are standing together at the corner of a rectangul
```
0.75(s14) → 0.50(s35) → 0.62(s62) → 0.75(s90) → ... → 0.12(s448) → 0.38(s470)
```

**Q:** /no_think If $x^3$ is a positive factor of $10!,$ how many possible integer
```
1.00(s1) → 1.00(s41) → 1.00(s65) → 1.00(s79) → ... → 0.88(s453) → 0.75(s480)
```

**Q:** /no_think Two numbers, $x$ and $y$ are selected at random from the interval
```
0.12(s21) → 0.00(s34) → 0.25(s56) → 0.12(s80) → ... → 0.00(s448) → 0.00(s470)
```

**Q:** /no_think If \[f(n + 1) = (-1)^{n + 1} n - 2f(n)\]for $n \ge 1,$ and $f(1) 
```
0.50(s24) → 0.38(s32) → 0.25(s51) → 0.12(s87) → ... → 0.12(s450) → 0.25(s469)
```

**Q:** /no_think The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$ and $x^2 - y^2 + 6x
```
0.25(s31) → 0.25(s53) → 0.00(s91) → 0.50(s114) → ... → 0.25(s435) → 0.12(s470)
```

**Q:** /no_think The two-digit number $``B6,''$ where $B$ is the tens digit, is th
```
0.88(s17) → 0.62(s30) → 0.62(s69) → 1.00(s86) → ... → 0.38(s438) → 0.75(s475)
```

**Q:** /no_think Convert the point $(0,3)$ in rectangular coordinates to polar coo
```
1.00(s41) → 1.00(s56) → 1.00(s89) → 1.00(s119) → ... → 1.00(s434) → 0.75(s479)
```

**Q:** /no_think Steve says to Jon, "I am thinking of a polynomial whose roots are
```
0.25(s7) → 0.00(s40) → 0.12(s67) → 0.12(s106) → ... → 0.00(s443) → 0.00(s466)
```

**Q:** /no_think If $f(x)=5x^2+3x+4$, what is the value of $f(-2)$? Let's think st
```
1.00(s1) → 1.00(s36) → 1.00(s51) → 1.00(s79) → ... → 0.88(s456) → 0.88(s466)
```

**Q:** /no_think Determine the modulo 4 remainder of the following sum: $$ 1 + 2 +
```
1.00(s13) → 1.00(s44) → 1.00(s64) → 1.00(s92) → ... → 1.00(s439) → 0.88(s471)
```

**Q:** /no_think Let $a,$ $b,$ and $c$ be distinct real numbers.  Find the degree 
```
0.38(s19) → 0.25(s48) → 0.38(s59) → 0.25(s76) → ... → 0.25(s449) → 0.25(s470)
```

**Q:** /no_think Find all $p$ which satisfy both the inequalities $0\ge 54p-144$ a
```
1.00(s16) → 1.00(s26) → 1.00(s78) → 1.00(s110) → ... → 1.00(s434) → 0.88(s469)
```

**Q:** /no_think In the diagram below, we have $\overline{ST}\parallel\overline{QR
```
0.12(s9) → 0.00(s37) → 0.00(s51) → 0.00(s91) → ... → 0.00(s437) → 0.00(s472)
```


## Endpoint 退化题完整列表（40 道）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | half-avg差 | 真退化 |
|-------------|-----|--------|--------|------|-----------|--------|
| /no_think If $re^{i \theta}$ is a root of \[z^8 - z^7 +... | `8 \pi` | 0.625 | 0.000 | -0.625 | -0.104 | ✓ |
| /no_think What is the range of the function $y=\log_2 (... | `(-\infty, 0]` | 0.500 | 0.000 | -0.500 | -0.153 | ✓ |
| /no_think In the circle with center $Q$, radii $AQ$ and... | `\frac{14}{3}` | 0.375 | 0.000 | -0.375 | -0.112 | ✓ |
| /no_think A regular hexagon can be divided into six equ... | `42` | 1.000 | 0.625 | -0.375 | +0.214 |  |
| /no_think Jim and Martha are standing together at the c... | `200` | 0.750 | 0.375 | -0.375 | -0.069 | ✓ |
| /no_think Two-thirds of the students at Baker Middle Sc... | `1251` | 1.000 | 0.625 | -0.375 | -0.085 | ✓ |
| /no_think If $x^3$ is a positive factor of $10!,$ how m... | `6` | 1.000 | 0.750 | -0.250 | -0.014 | ✓ |
| /no_think Two runners, $A$ and $B,$ start at a point $O... | `30^\circ` | 0.750 | 0.500 | -0.250 | +0.012 |  |
| /no_think Steve says to Jon, "I am thinking of a polyno... | `440` | 0.250 | 0.000 | -0.250 | -0.050 | ✓ |
| /no_think Let $f(x)=\|x-p\|+\|x-15\|+\|x-p-15\|,$ where $0 < ... | `15` | 1.000 | 0.750 | -0.250 | +0.000 |  |
| /no_think If \[f(n + 1) = (-1)^{n + 1} n - 2f(n)\]for $... | `331` | 0.500 | 0.250 | -0.250 | -0.013 | ✓ |
| /no_think Convert the point $(0,3)$ in rectangular coor... | `\left( 3, \frac` | 1.000 | 0.750 | -0.250 | -0.028 | ✓ |
| /no_think If $f(x)=5x^2+3x+4$, what is the value of $f(... | `18` | 1.000 | 0.875 | -0.125 | -0.025 | ✓ |
| /no_think What is the sum of the digits in the terminat... | `14` | 0.625 | 0.500 | -0.125 | +0.092 |  |
| /no_think If no one shares an office, in how many ways ... | `60` | 1.000 | 0.875 | -0.125 | +0.029 |  |
| /no_think If $0.\overline{1331}$ is written as a fracti... | `1030` | 1.000 | 0.875 | -0.125 | +0.147 |  |
| /no_think $n$ fair 6-sided dice are simultaneously roll... | `4` | 1.000 | 0.875 | -0.125 | +0.000 |  |
| /no_think In the diagram, $D$ and $E$ are the midpoints... | `8` | 1.000 | 0.875 | -0.125 | +0.017 |  |
| /no_think Let $f(x) = x^3 + 3x^2 + 1.$  There exist rea... | `(-2,1)` | 1.000 | 0.875 | -0.125 | -0.022 |  |
| /no_think Twelve 1 by 1 squares form a rectangle, as sh... | `10` | 1.000 | 0.875 | -0.125 | +0.137 |  |
| /no_think Let $f(x)=\left\lfloor\left(-\frac58\right)^x... | `3` | 0.625 | 0.500 | -0.125 | +0.028 |  |
| /no_think What is the number of square units in the are... | `12` | 1.000 | 0.875 | -0.125 | +0.029 |  |
| /no_think In the diagram below, we have $\overline{ST}\... | `75^\circ` | 0.125 | 0.000 | -0.125 | -0.013 | ✓ |
| /no_think In how many ways can $7$ people sit around a ... | `144` | 0.250 | 0.125 | -0.125 | +0.028 |  |
| /no_think In a convex quadrilateral, the measure of the... | `120` | 0.750 | 0.625 | -0.125 | +0.060 |  |
| /no_think Find the degree measure of the least positive... | `120^\circ` | 1.000 | 0.875 | -0.125 | +0.042 |  |
| /no_think Determine the modulo 4 remainder of the follo... | `2` | 1.000 | 0.875 | -0.125 | -0.028 | ✓ |
| /no_think Alice and Bob are playing a game. Alice start... | `\frac{2}{3}` | 1.000 | 0.875 | -0.125 | +0.025 |  |
| /no_think $\overline{BC}$ is parallel to the segment th... | `28` | 0.500 | 0.375 | -0.125 | -0.225 | ✓ |
| /no_think Find all $p$ which satisfy both the inequalit... | `\left(\frac{3}{` | 1.000 | 0.875 | -0.125 | -0.012 | ✓ |
| /no_think A book is said to have $n$ leaves if it is co... | `103` | 0.875 | 0.750 | -0.125 | +0.072 |  |
| /no_think The two-digit number $``B6,''$ where $B$ is t... | `2` | 0.875 | 0.750 | -0.125 | -0.075 | ✓ |
| /no_think How many sides would there be in a convex pol... | `8` | 1.000 | 0.875 | -0.125 | +0.000 |  |
| /no_think Let $a,$ $b,$ and $c$ be distinct real number... | `0` | 0.375 | 0.250 | -0.125 | -0.049 | ✓ |
| /no_think Two numbers, $x$ and $y$ are selected at rand... | `\frac{1}{2}` | 0.125 | 0.000 | -0.125 | -0.071 | ✓ |
| /no_think What is the smallest positive integer $n$ suc... | `6` | 1.000 | 0.875 | -0.125 | +0.025 |  |
| /no_think The Greek army contained two types of soldier... | `225` | 1.000 | 0.875 | -0.125 | +0.018 |  |
| /no_think The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$... | `40` | 0.250 | 0.125 | -0.125 | -0.026 | ✓ |
| /no_think Linda, Sherry, June, and Connie walked around... | `59` | 0.375 | 0.250 | -0.125 | +0.092 |  |
| /no_think Let $a$ be a positive real number such that a... | `3` | 0.750 | 0.625 | -0.125 | +0.164 |  |

## 高波动 Top-10 题目（policy 最不稳定）

| 题目（截断） | mean_acc | 波动率 | diff |
|-------------|---------|--------|------|
| /no_think If $A$ is the sum of the positive divisors of... | 0.671 | 0.400 | +0.750 |
| /no_think If $a$ and $b$ are positive integers such tha... | 0.533 | 0.356 | +0.875 |
| /no_think What is the smallest number which is one less... | 0.800 | 0.299 | +0.375 |
| /no_think Simplify $\tan 100^\circ + 4 \sin 100^\circ.$... | 0.395 | 0.289 | +1.000 |
| /no_think Find the number of ordered pairs $(a,b)$ of i... | 0.586 | 0.283 | +0.250 |
| /no_think The set of points $(x,y,z)$ that satisfy \[2x... | 0.645 | 0.280 | +0.250 |
| /no_think Expand and simplify completely: \begin{align*... | 0.469 | 0.278 | +0.500 |
| /no_think Simplify \[\frac{\sec x}{\sin x} - \frac{\sin... | 0.632 | 0.275 | +0.750 |
| /no_think Compute $\sin^3 18^\circ + \sin^2 18^\circ.$ ... | 0.400 | 0.262 | +0.000 |
| /no_think Find the greatest common divisor of $3339$, $... | 0.382 | 0.255 | +0.125 |