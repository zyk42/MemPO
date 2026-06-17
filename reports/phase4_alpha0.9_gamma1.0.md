# Rollout 分析报告

**目录：** `rollout_logs_phase4_alpha0.9_gamma1.0`
**训练步数：** 480（step 1 → 480）
**唯一题目数：** 398
**每题平均出现次数：** 19.3

## 退化/提升汇总

| 类别 | 题目数 | 占比 |
|------|--------|------|
| Endpoint 退化（last < first） | 29 | 7.3% |
| 真正退化（3指标全负） | 14 | 3.5% |
| 提升（last > first） | 134 | 33.7% |
| 不变 | 235 | 59.0% |

## Endpoint 退化幅度分布

| 幅度 | 题目数 |
|------|--------|
| <-0.5 | 1 |
| -0.5~-0.25 | 3 |
| -0.25~0 | 25 |

> 其中 **假退化**（endpoint 下降但 half-avg 未下降）：**15** 道，可能是末尾测量噪声，不代表真正遗忘。

## 波动率分析

| 分组 | 平均波动率（acc std across steps） |
|------|----------------------------------|
| Endpoint 退化题 | 0.1307 |
| 其他题目 | 0.0782 |

> 退化题波动率为稳定题的 **1.7x**，说明退化题本身 policy 不稳定。

## 真正退化的 14 道题（3指标全负）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | slope | 波动率 |
|-------------|-----|--------|--------|------|-------|--------|
| /no_think $\overline{BC}$ is parallel to the segment through... | `28` | 0.500 | 0.125 | -0.375 | -0.0261 | 0.193 |
| /no_think If $f(x)=5x^2+3x+4$, what is the value of $f(-2)$?... | `18` | 1.000 | 0.250 | -0.750 | -0.0173 | 0.182 |
| /no_think What is the range of the function $y=\log_2 (\sqrt... | `(-\infty, 0]` | 0.250 | 0.000 | -0.250 | -0.0138 | 0.150 |
| /no_think What is the least positive integer multiple of 30 ... | `2220` | 0.875 | 0.750 | -0.125 | -0.0101 | 0.128 |
| /no_think Linda, Sherry, June, and Connie walked around thei... | `59` | 0.250 | 0.000 | -0.250 | -0.0083 | 0.097 |
| /no_think Let $a,$ $b,$ and $c$ be distinct real numbers.  F... | `0` | 0.625 | 0.375 | -0.250 | -0.0081 | 0.171 |
| /no_think Six cars pull up to a red light, one at a time. At... | `540` | 0.750 | 0.375 | -0.375 | -0.0075 | 0.194 |
| /no_think Evaluate \[\sin (\arcsin 0.4 + \arcsin 0.5) \cdot ... | `\frac{9}{100}` | 0.250 | 0.125 | -0.125 | -0.0074 | 0.109 |
| /no_think Suppose $a$ and $b$ are different prime numbers gr... | `8` | 1.000 | 0.750 | -0.250 | -0.0066 | 0.070 |
| /no_think On a particular map, $3$ inches on the map equates... | `\frac{639}{40}` | 1.000 | 0.875 | -0.125 | -0.0058 | 0.115 |
| /no_think In the diagram, two circles, each with center $D$,... | `120` | 0.250 | 0.125 | -0.125 | -0.0055 | 0.121 |
| /no_think What is the number of square centimeters in the ar... | `21` | 0.125 | 0.000 | -0.125 | -0.0035 | 0.076 |
| /no_think An investment of $\$24,\!000$ is made in a governm... | `\$32,\!348` | 0.125 | 0.000 | -0.125 | -0.0031 | 0.039 |
| /no_think Find the curve defined by the equation \[r^2 \cos ... | `\text{(E)}` | 0.125 | 0.000 | -0.125 | -0.0021 | 0.038 |

### 真退化题轨迹

**Q:** /no_think $\overline{BC}$ is parallel to the segment through $A$, and $AB =
```
0.50(s16) → 0.62(s27) → 0.62(s72) → 0.50(s81) → ... → 0.00(s444) → 0.12(s465)
```

**Q:** /no_think If $f(x)=5x^2+3x+4$, what is the value of $f(-2)$? Let's think st
```
1.00(s1) → 1.00(s36) → 1.00(s51) → 1.00(s79) → ... → 1.00(s456) → 0.25(s466)
```

**Q:** /no_think What is the range of the function $y=\log_2 (\sqrt{\sin x})$ for 
```
0.25(s21) → 0.25(s39) → 0.50(s55) → 0.25(s74) → ... → 0.25(s441) → 0.00(s459)
```

**Q:** /no_think What is the least positive integer multiple of 30 that can be wri
```
0.88(s6) → 0.75(s28) → 0.88(s59) → 1.00(s80) → ... → 0.75(s443) → 0.75(s460)
```

**Q:** /no_think Linda, Sherry, June, and Connie walked around their neighborhoods
```
0.25(s31) → 0.12(s69) → 0.25(s78) → 0.12(s104) → ... → 0.00(s437) → 0.00(s464)
```

**Q:** /no_think Let $a,$ $b,$ and $c$ be distinct real numbers.  Find the degree 
```
0.62(s19) → 0.38(s48) → 0.62(s59) → 0.38(s76) → ... → 0.50(s449) → 0.38(s470)
```

**Q:** /no_think Six cars pull up to a red light, one at a time. At the light, the
```
0.75(s13) → 0.88(s25) → 0.75(s71) → 0.62(s77) → ... → 0.25(s443) → 0.38(s461)
```

**Q:** /no_think Evaluate \[\sin (\arcsin 0.4 + \arcsin 0.5) \cdot \sin (\arcsin 0
```
0.25(s11) → 0.25(s35) → 0.38(s54) → 0.12(s77) → ... → 0.00(s438) → 0.12(s473)
```

**Q:** /no_think Suppose $a$ and $b$ are different prime numbers greater than 2. H
```
1.00(s3) → 1.00(s40) → 1.00(s70) → 1.00(s86) → ... → 1.00(s414) → 0.75(s465)
```

**Q:** /no_think On a particular map, $3$ inches on the map equates to $10$ miles 
```
1.00(s10) → 1.00(s39) → 1.00(s57) → 0.88(s85) → ... → 1.00(s455) → 0.88(s464)
```

**Q:** /no_think In the diagram, two circles, each with center $D$, have radii of 
```
0.25(s17) → 0.12(s56) → 0.12(s76) → 0.00(s102) → ... → 0.00(s451) → 0.12(s457)
```

**Q:** /no_think What is the number of square centimeters in the area of this trap
```
0.12(s18) → 0.00(s35) → 0.12(s60) → 0.00(s95) → ... → 0.00(s456) → 0.00(s475)
```

**Q:** /no_think An investment of $\$24,\!000$ is made in a government bond that w
```
0.12(s10) → 0.00(s35) → 0.00(s53) → 0.00(s91) → ... → 0.00(s439) → 0.00(s472)
```

**Q:** /no_think Find the curve defined by the equation \[r^2 \cos 2 \theta = 4.\]
```
0.12(s23) → 0.00(s27) → 0.00(s69) → 0.00(s76) → ... → 0.00(s437) → 0.00(s461)
```


## Endpoint 退化题完整列表（29 道）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | half-avg差 | 真退化 |
|-------------|-----|--------|--------|------|-----------|--------|
| /no_think If $f(x)=5x^2+3x+4$, what is the value of $f(... | `18` | 1.000 | 0.250 | -0.750 | -0.175 | ✓ |
| /no_think Six cars pull up to a red light, one at a tim... | `540` | 0.750 | 0.375 | -0.375 | -0.025 | ✓ |
| /no_think $\overline{BC}$ is parallel to the segment th... | `28` | 0.500 | 0.125 | -0.375 | -0.275 | ✓ |
| /no_think What integer $n$ satisfies $0\le n<18$ and $$... | `13` | 0.500 | 0.125 | -0.375 | +0.150 |  |
| /no_think Suppose $a$ and $b$ are different prime numbe... | `8` | 1.000 | 0.750 | -0.250 | -0.062 | ✓ |
| /no_think Jim and Martha are standing together at the c... | `200` | 0.875 | 0.625 | -0.250 | +0.000 |  |
| /no_think Let $a,$ $b,$ and $c$ be distinct real number... | `0` | 0.625 | 0.375 | -0.250 | -0.104 | ✓ |
| /no_think What is the range of the function $y=\log_2 (... | `(-\infty, 0]` | 0.250 | 0.000 | -0.250 | -0.125 | ✓ |
| /no_think Linda, Sherry, June, and Connie walked around... | `59` | 0.250 | 0.000 | -0.250 | -0.092 | ✓ |
| /no_think If $x^3$ is a positive factor of $10!,$ how m... | `6` | 1.000 | 0.875 | -0.125 | -0.028 |  |
| /no_think What is the least positive integer multiple o... | `2220` | 0.875 | 0.750 | -0.125 | -0.086 | ✓ |
| /no_think What power of 4 is equal to 8? Express your a... | `\frac{3}{2}` | 1.000 | 0.875 | -0.125 | +0.017 |  |
| /no_think On a particular map, $3$ inches on the map eq... | `\frac{639}{40}` | 1.000 | 0.875 | -0.125 | -0.100 | ✓ |
| /no_think An investment of $\$24,\!000$ is made in a go... | `\$32,\!348` | 0.125 | 0.000 | -0.125 | -0.028 | ✓ |
| /no_think Evaluate \[\sin (\arcsin 0.4 + \arcsin 0.5) \... | `\frac{9}{100}` | 0.250 | 0.125 | -0.125 | -0.062 | ✓ |
| /no_think Compute $17^{-1}\pmod{83}$. Express your answ... | `44` | 0.625 | 0.500 | -0.125 | +0.113 |  |
| /no_think Find the smallest positive real number $C$ fo... | `4` | 0.375 | 0.250 | -0.125 | +0.103 |  |
| /no_think Find $\sin 20^\circ \sin 40^\circ \sin 60^\ci... | `\frac{9}{256}` | 0.375 | 0.250 | -0.125 | +0.150 |  |
| /no_think In the diagram, two circles, each with center... | `120` | 0.250 | 0.125 | -0.125 | -0.040 | ✓ |
| /no_think What is the number of square centimeters in t... | `21` | 0.125 | 0.000 | -0.125 | -0.019 | ✓ |
| /no_think Consider the function $z(x,y)$ describing the... | `-\frac{3}{8}` | 0.750 | 0.625 | -0.125 | +0.139 |  |
| /no_think Simplify \[\cos \left( \frac{2 \pi}{15} \righ... | `\frac{1}{16}` | 0.625 | 0.500 | -0.125 | +0.062 |  |
| /no_think Let $F_1$ and $F_2$ be the foci of the ellips... | `2` | 0.875 | 0.750 | -0.125 | -0.043 |  |
| /no_think The number $(\sqrt{2}+\sqrt{3})^3$ can be wri... | `20` | 1.000 | 0.875 | -0.125 | +0.001 |  |
| /no_think If $\frac{a}{b}$ is the probability that the ... | `202` | 0.500 | 0.375 | -0.125 | +0.069 |  |
| /no_think Let $F(z)=\frac{z+i}{z-i}$ for all complex nu... | `1+274i` | 0.125 | 0.000 | -0.125 | +0.025 |  |
| /no_think Find the curve defined by the equation \[r^2 ... | `\text{(E)}` | 0.125 | 0.000 | -0.125 | -0.025 | ✓ |
| /no_think If \[f(n + 1) = (-1)^{n + 1} n - 2f(n)\]for $... | `331` | 0.375 | 0.250 | -0.125 | -0.003 |  |
| /no_think If \[\sqrt[3]{2} = a + \cfrac{1}{b + \cfrac{1... | `3` | 0.875 | 0.750 | -0.125 | +0.042 |  |

## 高波动 Top-10 题目（policy 最不稳定）

| 题目（截断） | mean_acc | 波动率 | diff |
|-------------|---------|--------|------|
| /no_think If $A$ is the sum of the positive divisors of... | 0.651 | 0.345 | +0.750 |
| /no_think What is the value of $$ (3x-2)(4x+1)-(3x-2)4x... | 0.394 | 0.342 | +1.000 |
| /no_think What is the result when the greatest common f... | 0.526 | 0.337 | +0.625 |
| /no_think Expand and simplify completely: \begin{align*... | 0.500 | 0.281 | +0.500 |
| /no_think Find the greatest integer less than $(\sqrt{7... | 0.428 | 0.277 | +0.750 |
| /no_think Given that \begin{align*}x_{1}&=211,\\ x_{2}&... | 0.156 | 0.275 | +0.625 |
| /no_think Simplify $\tan 100^\circ + 4 \sin 100^\circ.$... | 0.493 | 0.272 | +0.500 |
| /no_think Recently, Frank took a one-hundred question a... | 0.844 | 0.269 | +0.000 |
| /no_think For how many real values of $x$ is $\sqrt{120... | 0.700 | 0.267 | +0.500 |
| /no_think Let $a,$ $b,$ and $c$ be positive real number... | 0.606 | 0.254 | +0.625 |