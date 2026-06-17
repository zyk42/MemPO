# Rollout 分析报告

**目录：** `rollout_logs_phase3_filter_only_alpha0.9_gamma0.0`
**训练步数：** 480（step 1 → 480）
**唯一题目数：** 398
**每题平均出现次数：** 19.3

## 退化/提升汇总

| 类别 | 题目数 | 占比 |
|------|--------|------|
| Endpoint 退化（last < first） | 24 | 6.0% |
| 真正退化（3指标全负） | 11 | 2.8% |
| 提升（last > first） | 155 | 38.9% |
| 不变 | 219 | 55.0% |

## Endpoint 退化幅度分布

| 幅度 | 题目数 |
|------|--------|
| <-0.5 | 2 |
| -0.5~-0.25 | 0 |
| -0.25~0 | 22 |

> 其中 **假退化**（endpoint 下降但 half-avg 未下降）：**13** 道，可能是末尾测量噪声，不代表真正遗忘。

## 波动率分析

| 分组 | 平均波动率（acc std across steps） |
|------|----------------------------------|
| Endpoint 退化题 | 0.1188 |
| 其他题目 | 0.0865 |

> 退化题波动率为稳定题的 **1.4x**，说明退化题本身 policy 不稳定。

## 真正退化的 11 道题（3指标全负）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | slope | 波动率 |
|-------------|-----|--------|--------|------|-------|--------|
| /no_think Let $F_1$ and $F_2$ be the foci of the ellipse $kx... | `2` | 0.750 | 0.125 | -0.625 | -0.0246 | 0.238 |
| /no_think $\overline{BC}$ is parallel to the segment through... | `28` | 0.250 | 0.000 | -0.250 | -0.0212 | 0.247 |
| /no_think Linda, Sherry, June, and Connie walked around thei... | `59` | 0.625 | 0.000 | -0.625 | -0.0186 | 0.168 |
| /no_think In the diagram, two circles, each with center $D$,... | `120` | 0.375 | 0.125 | -0.250 | -0.0140 | 0.164 |
| /no_think The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$ and ... | `40` | 0.125 | 0.000 | -0.125 | -0.0136 | 0.122 |
| /no_think For $0 \le x \le 40$ and $0 \le y \le 50,$ find th... | `70 \sqrt{2}` | 0.125 | 0.000 | -0.125 | -0.0056 | 0.140 |
| /no_think What is the number of square centimeters in the ar... | `21` | 0.125 | 0.000 | -0.125 | -0.0055 | 0.087 |
| /no_think How many ways are there to put 5 balls in 2 boxes ... | `3` | 1.000 | 0.750 | -0.250 | -0.0047 | 0.095 |
| /no_think Jim and Martha are standing together at the corner... | `200` | 0.625 | 0.375 | -0.250 | -0.0041 | 0.194 |
| /no_think In the figure below, quadrilateral $CDEG$ is a squ... | `1\frac{4}{5}` | 0.125 | 0.000 | -0.125 | -0.0041 | 0.040 |
| /no_think Zach has three bags and a bunch of pencils to be p... | `2` | 1.000 | 0.875 | -0.125 | -0.0027 | 0.046 |

### 真退化题轨迹

**Q:** /no_think Let $F_1$ and $F_2$ be the foci of the ellipse $kx^2 + y^2 = 1,$ 
```
0.75(s18) → 0.75(s25) → 0.50(s87) → 1.00(s118) → ... → 0.62(s443) → 0.12(s473)
```

**Q:** /no_think $\overline{BC}$ is parallel to the segment through $A$, and $AB =
```
0.25(s16) → 0.50(s27) → 0.00(s72) → 0.88(s81) → ... → 0.25(s444) → 0.00(s465)
```

**Q:** /no_think Linda, Sherry, June, and Connie walked around their neighborhoods
```
0.62(s31) → 0.50(s69) → 0.12(s78) → 0.12(s104) → ... → 0.00(s437) → 0.00(s464)
```

**Q:** /no_think In the diagram, two circles, each with center $D$, have radii of 
```
0.38(s17) → 0.00(s56) → 0.62(s76) → 0.00(s102) → ... → 0.00(s451) → 0.12(s457)
```

**Q:** /no_think The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$ and $x^2 - y^2 + 6x
```
0.12(s31) → 0.25(s53) → 0.38(s91) → 0.25(s114) → ... → 0.12(s435) → 0.00(s470)
```

**Q:** /no_think For $0 \le x \le 40$ and $0 \le y \le 50,$ find the minimum value
```
0.12(s17) → 0.12(s31) → 0.25(s54) → 0.00(s82) → ... → 0.00(s444) → 0.00(s478)
```

**Q:** /no_think What is the number of square centimeters in the area of this trap
```
0.12(s18) → 0.00(s35) → 0.00(s60) → 0.25(s95) → ... → 0.00(s456) → 0.00(s475)
```

**Q:** /no_think How many ways are there to put 5 balls in 2 boxes if the balls ar
```
1.00(s18) → 1.00(s47) → 1.00(s59) → 1.00(s88) → ... → 0.88(s452) → 0.75(s464)
```

**Q:** /no_think Jim and Martha are standing together at the corner of a rectangul
```
0.62(s14) → 0.62(s35) → 0.88(s62) → 0.88(s90) → ... → 1.00(s448) → 0.38(s470)
```

**Q:** /no_think In the figure below, quadrilateral $CDEG$ is a square with $CD = 
```
0.12(s19) → 0.12(s35) → 0.00(s68) → 0.00(s104) → ... → 0.00(s436) → 0.00(s458)
```

**Q:** /no_think Zach has three bags and a bunch of pencils to be placed into the 
```
1.00(s20) → 1.00(s30) → 1.00(s70) → 1.00(s78) → ... → 1.00(s448) → 0.88(s470)
```


## Endpoint 退化题完整列表（24 道）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | half-avg差 | 真退化 |
|-------------|-----|--------|--------|------|-----------|--------|
| /no_think Let $F_1$ and $F_2$ be the foci of the ellips... | `2` | 0.750 | 0.125 | -0.625 | -0.267 | ✓ |
| /no_think Linda, Sherry, June, and Connie walked around... | `59` | 0.625 | 0.000 | -0.625 | -0.176 | ✓ |
| /no_think Point $A$ lies somewhere within or on the squ... | `\frac{3}{2}` | 0.375 | 0.125 | -0.250 | +0.075 |  |
| /no_think For how many real values of $x$ is $\sqrt{120... | `11` | 0.750 | 0.500 | -0.250 | +0.088 |  |
| /no_think Jim and Martha are standing together at the c... | `200` | 0.625 | 0.375 | -0.250 | -0.083 | ✓ |
| /no_think $\overline{BC}$ is parallel to the segment th... | `28` | 0.250 | 0.000 | -0.250 | -0.325 | ✓ |
| /no_think In the diagram, two circles, each with center... | `120` | 0.375 | 0.125 | -0.250 | -0.144 | ✓ |
| /no_think How many ways are there to put 5 balls in 2 b... | `3` | 1.000 | 0.750 | -0.250 | -0.025 | ✓ |
| /no_think How many sides would there be in a convex pol... | `8` | 1.000 | 0.750 | -0.250 | +0.014 |  |
| /no_think Let $\mathbf{a},$ $\mathbf{b},$ $\mathbf{c}$ ... | `\begin{pmatrix}` | 1.000 | 0.875 | -0.125 | +0.101 |  |
| /no_think Below is a magic square, meaning that the sum... | `7` | 0.750 | 0.625 | -0.125 | +0.094 |  |
| /no_think What is the domain of the function $f(x) = \f... | `(2,12) \cup (12` | 0.875 | 0.750 | -0.125 | +0.150 |  |
| /no_think Tom got a Mr. Potato Head for his birthday. I... | `64` | 0.125 | 0.000 | -0.125 | +0.013 |  |
| /no_think Suzanne walks four miles every third day. Wha... | `36` | 1.000 | 0.875 | -0.125 | +0.012 |  |
| /no_think The sum of 27 consecutive positive integers i... | `81` | 1.000 | 0.875 | -0.125 | +0.000 |  |
| /no_think Consider the rectangular region with the foll... | `63` | 0.750 | 0.625 | -0.125 | +0.007 |  |
| /no_think For $0 \le x \le 40$ and $0 \le y \le 50,$ fi... | `70 \sqrt{2}` | 0.125 | 0.000 | -0.125 | -0.050 | ✓ |
| /no_think What is the number of square centimeters in t... | `21` | 0.125 | 0.000 | -0.125 | -0.047 | ✓ |
| /no_think In the figure below, quadrilateral $CDEG$ is ... | `1\frac{4}{5}` | 0.125 | 0.000 | -0.125 | -0.028 | ✓ |
| /no_think Zach has three bags and a bunch of pencils to... | `2` | 1.000 | 0.875 | -0.125 | -0.013 | ✓ |
| /no_think For a point $P,$ let $d_1,$ $d_2$ and $d_3$ r... | `288 \pi` | 0.125 | 0.000 | -0.125 | -0.012 |  |
| /no_think Solve \[\frac{1}{x - 5} > 0.\]Enter your answ... | `(5,\infty)` | 1.000 | 0.875 | -0.125 | +0.069 |  |
| /no_think Two candidates, Dan and Donald, run for class... | `\frac14` | 1.000 | 0.875 | -0.125 | +0.025 |  |
| /no_think The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$... | `40` | 0.125 | 0.000 | -0.125 | -0.135 | ✓ |

## 高波动 Top-10 题目（policy 最不稳定）

| 题目（截断） | mean_acc | 波动率 | diff |
|-------------|---------|--------|------|
| /no_think Given that \begin{align*}x_{1}&=211,\\ x_{2}&... | 0.312 | 0.403 | +0.875 |
| /no_think What is the value of $$ (3x-2)(4x+1)-(3x-2)4x... | 0.594 | 0.376 | +1.000 |
| /no_think Simplify $\tan 100^\circ + 4 \sin 100^\circ.$... | 0.559 | 0.345 | +0.875 |
| /no_think Let $a,$ $b,$ $c,$ $d$ be positive real numbe... | 0.637 | 0.334 | +0.750 |
| /no_think If $A$ is the sum of the positive divisors of... | 0.704 | 0.334 | +0.625 |
| /no_think The polynomial $x^3 - 3x^2 + 4x - 1$ is a fac... | 0.250 | 0.315 | +1.000 |
| /no_think The set of points $(x,y,z)$ that satisfy \[2x... | 0.711 | 0.303 | +0.375 |
| /no_think Let $z$ be a complex number such that \[z + \... | 0.500 | 0.298 | +0.375 |
| /no_think Let $a,$ $b,$ and $c$ be positive real number... | 0.631 | 0.294 | +0.875 |
| /no_think What is the smallest number which is one less... | 0.756 | 0.282 | +0.750 |