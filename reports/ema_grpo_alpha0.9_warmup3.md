# Rollout 分析报告

**目录：** `rollout_logs_alpha0.9_warmup3`
**训练步数：** 165（step 1 → 165）
**唯一题目数：** 398
**每题平均出现次数：** 6.6

## 退化/提升汇总

| 类别 | 题目数 | 占比 |
|------|--------|------|
| Endpoint 退化（last < first） | 37 | 9.3% |
| 真正退化（3指标全负） | 26 | 6.5% |
| 提升（last > first） | 54 | 13.6% |
| 不变 | 307 | 77.1% |

## Endpoint 退化幅度分布

| 幅度 | 题目数 |
|------|--------|
| <-0.5 | 1 |
| -0.5~-0.25 | 4 |
| -0.25~0 | 32 |

> 其中 **假退化**（endpoint 下降但 half-avg 未下降）：**11** 道，可能是末尾测量噪声，不代表真正遗忘。

## 波动率分析

| 分组 | 平均波动率（acc std across steps） |
|------|----------------------------------|
| Endpoint 退化题 | 0.1344 |
| 其他题目 | 0.0335 |

> 退化题波动率为稳定题的 **4.0x**，说明退化题本身 policy 不稳定。

## 真正退化的 26 道题（3指标全负）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | slope | 波动率 |
|-------------|-----|--------|--------|------|-------|--------|
| Consider the rectangular region with the following points as... | `63` | 1.000 | 0.375 | -0.625 | -0.0893 | 0.215 |
| Steve says to Jon, "I am thinking of a polynomial whose root... | `440` | 0.625 | 0.250 | -0.375 | -0.0750 | 0.153 |
| Linda, Sherry, June, and Connie walked around their neighbor... | `59` | 0.875 | 0.625 | -0.250 | -0.0536 | 0.153 |
| Let $f(x) = x^{10}+5x^9-8x^8+7x^7-x^6-12x^5+4x^4-8x^3+12x^2-... | `-13x+3` | 0.875 | 0.750 | -0.125 | -0.0500 | 0.105 |
| A $90^\circ$ rotation around $-4 - 5i$ in the clockwise dire... | `1 - 12i` | 1.000 | 0.500 | -0.500 | -0.0491 | 0.187 |
| Each week, between 30 and 50 students show up for an archery... | `41` | 0.625 | 0.500 | -0.125 | -0.0393 | 0.200 |
| If $\frac{a}{b}$ is the probability that the reciprocal of a... | `202` | 1.000 | 0.750 | -0.250 | -0.0393 | 0.105 |
| Let $p(x)$ be a polynomial of degree 5 such that \[p(n) = \f... | `\frac{3}{56}` | 0.125 | 0.000 | -0.125 | -0.0393 | 0.094 |
| The symbols $\triangle$, $\square$, $\diamond$, $\clubsuit$ ... | `3` | 1.000 | 0.875 | -0.125 | -0.0357 | 0.102 |
| Find the smallest positive real number $C$ for which \[\left... | `4` | 0.875 | 0.625 | -0.250 | -0.0357 | 0.102 |
| The Greek army contained two types of soldiers: the upper cl... | `225` | 1.000 | 0.875 | -0.125 | -0.0357 | 0.102 |
| Let $a$ be a positive real number such that all the roots of... | `3` | 0.875 | 0.750 | -0.125 | -0.0357 | 0.129 |
| Bob and Alice each have a bag that contains one ball of each... | `\frac{1}{3}` | 0.375 | 0.000 | -0.375 | -0.0321 | 0.184 |
| In $\triangle{RST}$, shown, $\sin{R}=\frac{2}{5}$.  What is ... | `\frac{\sqrt{21}}{5}` | 1.000 | 0.875 | -0.125 | -0.0321 | 0.068 |
| If each point of the circle $x^2 + y^2 = 25$ is reflected in... | `(1,-16,-4,43)` | 1.000 | 0.875 | -0.125 | -0.0312 | 0.086 |
| Twelve friends met for dinner at Oscar's Overstuffed Oyster ... | `8` | 0.250 | 0.125 | -0.125 | -0.0312 | 0.134 |
| Let $\mathbf{a},$ $\mathbf{b},$ $\mathbf{c}$ be three vector... | `\begin{pmatrix} -18 ` | 1.000 | 0.875 | -0.125 | -0.0268 | 0.144 |
| When rolling a certain unfair six-sided die with faces numbe... | `29` | 0.875 | 0.625 | -0.250 | -0.0268 | 0.134 |
| Evaluate \[\sin (\arcsin 0.4 + \arcsin 0.5) \cdot \sin (\arc... | `\frac{9}{100}` | 0.625 | 0.375 | -0.250 | -0.0223 | 0.098 |
| For what real values of $x$ is $-4<x^{4}+4x^{2}<21$ satisfie... | `(-\sqrt{3}, \sqrt{3}` | 0.875 | 0.750 | -0.125 | -0.0179 | 0.159 |
| In a certain isosceles right triangle, the altitude to the h... | `32` | 1.000 | 0.875 | -0.125 | -0.0179 | 0.051 |
| In circle $J$, $HO$ and $HN$ are tangent to the circle at $O... | `180^\circ` | 0.875 | 0.750 | -0.125 | -0.0134 | 0.161 |
| On a particular map, $3$ inches on the map equates to $10$ m... | `\frac{639}{40}` | 1.000 | 0.875 | -0.125 | -0.0134 | 0.047 |
| Express the quotient $413_5 \div 2_5$ in base 5. Let's think... | `204_5` | 0.125 | 0.000 | -0.125 | -0.0134 | 0.061 |
| For some value of $x,$ $0 < x < 180,$ \[\tan 53^\circ \tan 8... | `46` | 1.000 | 0.875 | -0.125 | -0.0134 | 0.067 |
| Altitudes $\overline{AD}$ and $\overline{BE}$ of $\triangle ... | `106^\circ` | 0.875 | 0.750 | -0.125 | -0.0089 | 0.152 |

### 真退化题轨迹

**Q:** Consider the rectangular region with the following points as vertices: $$(5
```
1.00(s13) → 0.75(s37) → 0.88(s53) → 0.88(s76) → 0.75(s110) → 0.38(s141)
```

**Q:** Steve says to Jon, "I am thinking of a polynomial whose roots are all posit
```
0.62(s7) → 0.62(s40) → 0.38(s67) → 0.38(s106) → 0.38(s130) → 0.25(s146)
```

**Q:** Linda, Sherry, June, and Connie walked around their neighborhoods selling g
```
0.88(s31) → 1.00(s69) → 0.88(s78) → 0.62(s104) → 0.88(s135) → 0.62(s158)
```

**Q:** Let $f(x) = x^{10}+5x^9-8x^8+7x^7-x^6-12x^5+4x^4-8x^3+12x^2-5x-5$.  Without
```
0.88(s1) → 1.00(s70) → 0.88(s90) → 0.75(s117) → 0.75(s143)
```

**Q:** A $90^\circ$ rotation around $-4 - 5i$ in the clockwise direction is applie
```
1.00(s3) → 1.00(s30) → 0.88(s59) → 1.00(s83) → 1.00(s109) → 1.00(s121) → 0.50(s164)
```

**Q:** Each week, between 30 and 50 students show up for an archery class run by B
```
0.62(s9) → 1.00(s38) → 0.50(s68) → 0.50(s81) → 0.75(s113) → 0.50(s125)
```

**Q:** If $\frac{a}{b}$ is the probability that the reciprocal of a randomly selec
```
1.00(s22) → 1.00(s34) → 1.00(s55) → 0.88(s74) → 1.00(s105) → 0.75(s130)
```

**Q:** Let $p(x)$ be a polynomial of degree 5 such that \[p(n) = \frac{n}{n^2 - 1}
```
0.12(s24) → 0.25(s33) → 0.12(s67) → 0.12(s88) → 0.00(s118) → 0.00(s143)
```

**Q:** The symbols $\triangle$, $\square$, $\diamond$, $\clubsuit$ represent four 
```
1.00(s6) → 1.00(s42) → 0.88(s71) → 1.00(s85) → 0.75(s114) → 0.88(s143)
```

**Q:** Find the smallest positive real number $C$ for which \[\left\| \begin{pmatr
```
0.88(s15) → 0.75(s29) → 0.88(s61) → 0.88(s95) → 0.75(s115) → 0.62(s156)
```

**Q:** The Greek army contained two types of soldiers: the upper class and the low
```
1.00(s23) → 1.00(s25) → 0.88(s58) → 1.00(s113) → 0.75(s141) → 0.88(s162)
```

**Q:** Let $a$ be a positive real number such that all the roots of \[x^3 + ax^2 +
```
0.88(s36) → 0.75(s54) → 0.75(s94) → 0.50(s99) → 0.62(s139) → 0.75(s156)
```

**Q:** Bob and Alice each have a bag that contains one ball of each of the colors,
```
0.38(s13) → 0.25(s32) → 0.12(s58) → 0.12(s84) → 0.50(s111) → 0.00(s124)
```

**Q:** In $\triangle{RST}$, shown, $\sin{R}=\frac{2}{5}$.  What is $\sin{T}$?  [as
```
1.00(s14) → 1.00(s26) → 1.00(s59) → 0.88(s76) → 0.88(s102) → 0.88(s130)
```

**Q:** If each point of the circle $x^2 + y^2 = 25$ is reflected in the point $(4,
```
1.00(s7) → 1.00(s32) → 0.88(s56) → 0.88(s87) → 0.88(s108) → 0.75(s131) → 0.88(s157)
```

**Q:** Twelve friends met for dinner at Oscar's Overstuffed Oyster House, and each
```
0.25(s15) → 0.12(s38) → 0.38(s63) → 0.00(s74) → 0.12(s107) → 0.00(s131) → 0.12(s153)
```

**Q:** Let $\mathbf{a},$ $\mathbf{b},$ $\mathbf{c}$ be three vectors such that \[\
```
1.00(s2) → 0.75(s29) → 1.00(s54) → 1.00(s78) → 0.88(s103) → 0.62(s121) → 0.88(s154)
```

**Q:** When rolling a certain unfair six-sided die with faces numbered 1, 2, 3, 4,
```
0.88(s13) → 0.62(s42) → 1.00(s59) → 0.75(s73) → 0.75(s104) → 0.75(s142) → 0.62(s153)
```

**Q:** Evaluate \[\sin (\arcsin 0.4 + \arcsin 0.5) \cdot \sin (\arcsin 0.5 - \arcs
```
0.62(s11) → 0.62(s35) → 0.50(s54) → 0.50(s77) → 0.62(s112) → 0.62(s141) → 0.38(s165)
```

**Q:** For what real values of $x$ is $-4<x^{4}+4x^{2}<21$ satisfied? Express your
```
0.88(s14) → 0.62(s33) → 1.00(s55) → 1.00(s77) → 0.62(s115) → 0.75(s123) → 0.75(s146)
```

**Q:** In a certain isosceles right triangle, the altitude to the hypotenuse has l
```
1.00(s14) → 1.00(s32) → 1.00(s83) → 1.00(s114) → 1.00(s125) → 0.88(s148)
```

**Q:** In circle $J$, $HO$ and $HN$ are tangent to the circle at $O$ and $N$. Find
```
0.88(s4) → 0.50(s34) → 0.62(s52) → 0.62(s91) → 0.38(s103) → 0.62(s144) → 0.75(s154)
```

**Q:** On a particular map, $3$ inches on the map equates to $10$ miles in real li
```
1.00(s10) → 0.88(s39) → 0.88(s57) → 0.88(s85) → 0.88(s98) → 0.88(s127) → 0.88(s162)
```

**Q:** Express the quotient $413_5 \div 2_5$ in base 5. Let's think step by step a
```
0.12(s11) → 0.00(s39) → 0.00(s66) → 0.12(s93) → 0.00(s110) → 0.00(s137) → 0.00(s148)
```

**Q:** For some value of $x,$ $0 < x < 180,$ \[\tan 53^\circ \tan 81^\circ \tan x^
```
1.00(s23) → 1.00(s35) → 0.88(s70) → 1.00(s84) → 0.88(s105) → 1.00(s141) → 0.88(s152)
```

**Q:** Altitudes $\overline{AD}$ and $\overline{BE}$ of $\triangle ABC$ intersect 
```
0.88(s8) → 0.62(s36) → 0.88(s63) → 0.88(s74) → 0.50(s119) → 0.88(s133) → 0.75(s152)
```


## Endpoint 退化题完整列表（37 道）

| 题目（截断） | GTS | 首次acc | 末次acc | 差值 | half-avg差 | 真退化 |
|-------------|-----|--------|--------|------|-----------|--------|
| Consider the rectangular region with the following poin... | `63` | 1.000 | 0.375 | -0.625 | -0.208 | ✓ |
| A $90^\circ$ rotation around $-4 - 5i$ in the clockwise... | `1 - 12i` | 1.000 | 0.500 | -0.500 | -0.083 | ✓ |
| Steve says to Jon, "I am thinking of a polynomial whose... | `440` | 0.625 | 0.250 | -0.375 | -0.208 | ✓ |
| Bob and Alice each have a bag that contains one ball of... | `\frac{1}{3}` | 0.375 | 0.000 | -0.375 | -0.042 | ✓ |
| Let $F_1$ and $F_2$ be the foci of the ellipse $kx^2 + ... | `2` | 0.625 | 0.250 | -0.375 | +0.000 |  |
| In the diagram, four circles of radius 1 with centres $... | `30^\circ` | 0.500 | 0.250 | -0.250 | +0.073 |  |
| Two runners, $A$ and $B,$ start at a point $O$ on a lin... | `30^\circ` | 0.500 | 0.250 | -0.250 | +0.208 |  |
| A regular pentagon is rotated counterclockwise about it... | `72` | 0.875 | 0.625 | -0.250 | +0.000 |  |
| Evaluate \[\sin (\arcsin 0.4 + \arcsin 0.5) \cdot \sin ... | `\frac{9}{100}` | 0.625 | 0.375 | -0.250 | -0.052 | ✓ |
| When rolling a certain unfair six-sided die with faces ... | `29` | 0.875 | 0.625 | -0.250 | -0.115 | ✓ |
| Find the smallest positive real number $C$ for which \[... | `4` | 0.875 | 0.625 | -0.250 | -0.083 | ✓ |
| Let $z$ be a complex number such that \[z + \frac{1}{z}... | `-2` | 0.875 | 0.625 | -0.250 | +0.031 |  |
| In regular pentagon $FGHIJ$, extending the sides of the... | `36^\circ` | 0.750 | 0.500 | -0.250 | +0.083 |  |
| If $\frac{a}{b}$ is the probability that the reciprocal... | `202` | 1.000 | 0.750 | -0.250 | -0.125 | ✓ |
| Linda, Sherry, June, and Connie walked around their nei... | `59` | 0.875 | 0.625 | -0.250 | -0.208 | ✓ |
| Let $f(x) = x^{10}+5x^9-8x^8+7x^7-x^6-12x^5+4x^4-8x^3+1... | `-13x+3` | 0.875 | 0.750 | -0.125 | -0.146 | ✓ |
| Let $\mathbf{a},$ $\mathbf{b},$ $\mathbf{c}$ be three v... | `\begin{pmatrix}` | 1.000 | 0.875 | -0.125 | -0.073 | ✓ |
| In circle $J$, $HO$ and $HN$ are tangent to the circle ... | `180^\circ` | 0.875 | 0.750 | -0.125 | -0.073 | ✓ |
| The symbols $\triangle$, $\square$, $\diamond$, $\clubs... | `3` | 1.000 | 0.875 | -0.125 | -0.083 | ✓ |
| If each point of the circle $x^2 + y^2 = 25$ is reflect... | `(1,-16,-4,43)` | 1.000 | 0.875 | -0.125 | -0.115 | ✓ |
| Altitudes $\overline{AD}$ and $\overline{BE}$ of $\tria... | `106^\circ` | 0.875 | 0.750 | -0.125 | -0.042 | ✓ |
| Let $f(x)=\left\lfloor\left(-\frac58\right)^x\right\rfl... | `3` | 0.875 | 0.750 | -0.125 | +0.000 |  |
| Each week, between 30 and 50 students show up for an ar... | `41` | 0.625 | 0.500 | -0.125 | -0.125 | ✓ |
| On a particular map, $3$ inches on the map equates to $... | `\frac{639}{40}` | 1.000 | 0.875 | -0.125 | -0.042 | ✓ |
| Express the quotient $413_5 \div 2_5$ in base 5. Let's ... | `204_5` | 0.125 | 0.000 | -0.125 | -0.010 | ✓ |
| In $\triangle{RST}$, shown, $\sin{R}=\frac{2}{5}$.  Wha... | `\frac{\sqrt{21}` | 1.000 | 0.875 | -0.125 | -0.125 | ✓ |
| For what real values of $x$ is $-4<x^{4}+4x^{2}<21$ sat... | `(-\sqrt{3}, \sq` | 0.875 | 0.750 | -0.125 | -0.052 | ✓ |
| In a certain isosceles right triangle, the altitude to ... | `32` | 1.000 | 0.875 | -0.125 | -0.042 | ✓ |
| Twelve friends met for dinner at Oscar's Overstuffed Oy... | `8` | 0.250 | 0.125 | -0.125 | -0.188 | ✓ |
| Find $\sin 20^\circ \sin 40^\circ \sin 60^\circ \sin 80... | `\frac{9}{256}` | 0.750 | 0.625 | -0.125 | +0.083 |  |
| Let $\mathbf{A}$ be a matrix such that \[\mathbf{A} \be... | `\begin{pmatrix}` | 0.750 | 0.625 | -0.125 | +0.062 |  |
| Quadrilateral $ABCD$ is a square with area 16 square in... | `2` | 0.750 | 0.625 | -0.125 | +0.104 |  |
| Given that \begin{align*}x_{1}&=211,\\ x_{2}&=375,\\ x_... | `898` | 0.875 | 0.750 | -0.125 | +0.042 |  |
| The Greek army contained two types of soldiers: the upp... | `225` | 1.000 | 0.875 | -0.125 | -0.083 | ✓ |
| For some value of $x,$ $0 < x < 180,$ \[\tan 53^\circ \... | `46` | 1.000 | 0.875 | -0.125 | -0.021 | ✓ |
| Let $p(x)$ be a polynomial of degree 5 such that \[p(n)... | `\frac{3}{56}` | 0.125 | 0.000 | -0.125 | -0.125 | ✓ |
| Let $a$ be a positive real number such that all the roo... | `3` | 0.875 | 0.750 | -0.125 | -0.167 | ✓ |

## 高波动 Top-10 题目（policy 最不稳定）

| 题目（截断） | mean_acc | 波动率 | diff |
|-------------|---------|--------|------|
| Two runners, $A$ and $B,$ start at a point $O$ on a lin... | 0.411 | 0.320 | -0.250 |
| Compute $58_9 - 18_9.$ Express your answer in base $9.$... | 0.521 | 0.229 | +0.250 |
| The polynomial $x^3 - 3x^2 + 4x - 1$ is a factor of $x^... | 0.607 | 0.222 | +0.125 |
| If \[\sqrt[3]{2} = a + \cfrac{1}{b + \cfrac{1}{c + \cfr... | 0.458 | 0.219 | +0.500 |
| Consider the rectangular region with the following poin... | 0.771 | 0.215 | -0.625 |
| Solve \[\frac{\|x^2 - 81\|}{x^2 - 36x} < 0.\] Let's think... | 0.696 | 0.215 | +0.125 |
| Find the greatest integer less than $(\sqrt{7} + \sqrt{... | 0.536 | 0.213 | +0.250 |
| If $a$ and $b$ are positive integers such that $\gcd(a,... | 0.232 | 0.210 | +0.000 |
| Compute $\sin^3 18^\circ + \sin^2 18^\circ.$ Let's thin... | 0.661 | 0.200 | +0.000 |
| Each week, between 30 and 50 students show up for an ar... | 0.646 | 0.200 | -0.125 |