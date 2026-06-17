# EMA-GRPO：研究笔记

> 当前版本（去掉了已被证伪的方向和历史演化过程）

---

## 1. 问题背景

### 1.1 现象观察

在 Qwen3-1.7B × MATH-500 训练实验中发现：
- 373 道训练题中有 **16 道**从 epoch 1 到 epoch 9 出现了得分下降
- 典型退步模式：推导过程正确但最后一步认错目标量、反复验证循环偏离正确方向、边界条件丢失

### 1.2 标准 GRPO 的三个结构性缺陷

**缺陷一：归一化噪声**

标准 GRPO 用当前批次 n=8 rollout 估计归一化统计量：

$$A_i = \frac{r_i - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \varepsilon}$$

n=8 时均值标准误差 ≈ ±0.177（π=0.5），同一道题不同 step 的 advantage 基准相差很大，产生矛盾训练信号。

**缺陷二：零梯度保护被破坏**

标准 GRPO 在 8/8 全对时自然产生零梯度。若引入 EMA 均值但不加保护，EMA 均值滞后（如 μ̂=0.7），全对 rollout 的 advantage 仍为正，模型在已掌握题目上浪费梯度。

**缺陷三：截断响应的过度惩罚**

一个 batch 里同时包含简单题（score=1）和因 max_token 截断失败的困难题（score=0）。若批次均值=0.75：

$$A(\text{截断困难题}) = \frac{0 - 0.75}{\sigma} \approx -1.5 \sim -2.5$$

大负 advantage 直接惩罚"输出长推理链"的行为，导致 response_length 持续下降，最终模型失去深度推理能力（Qwen3-1.7B 实验中 AIME2024 pass@1 跌至 5%）。

---

## 2. 当前方法：EMA-GRPO + Mastery Filter

### 2.1 核心公式

$$A_i^{(p)} = r_i^{(p)} - \mu_{\text{EMA}}(p)$$

- 不使用 std 归一化（参考 Dr.GRPO：n=8 小样本 std 估计引入额外噪声，去掉反而更好）
- $\mu_{\text{EMA}}(p)$：prompt $p$ 历史奖励的指数移动平均

### 2.2 EMA 均值更新

$$\mu_t(p) = \alpha \cdot \mu_{t-1}(p) + (1-\alpha) \cdot \hat{\mu}_t(p)$$

- $\alpha = 0.9$，等效历史窗口 ≈ 10 步
- 首次见到 prompt（count=0）时：$\mu_1(p) = \hat{\mu}_t(p)$（cold-start 直接初始化，不从 0 起步）

### 2.3 掌握度过滤（Mastery Filter）

当前批次 8/8 全对时，强制回退批次统计：$A = r - 1 = 0$，恢复 GRPO 的零梯度保护。

完整 baseline 选择逻辑（按优先级）：

```
if batch_mean >= mastery_threshold (1.0):
    baseline = batch_mean      → A = r - 1 = 0（零梯度）
elif count >= warmup_steps (1):
    baseline = ema_mean        → A = r - μ_EMA（EMA 模式）
else:
    baseline = batch_mean      → A = r - μ_batch（warmup 回退）
```

### 2.4 预训练 μ₀ 初始化（Pre-init）

在训练开始前（`val_before_train` 之后、训练循环之前），对全体训练 prompt 运行一次 no-grad rollout pass，将 batch_mean 直接写入 memory bank 完成 μ₀ 初始化。

**效果**：所有 prompt 在训练开始时 count=1，warmup 保护立即解除，第一步训练直接使用 EMA。μ₀ 来自同一初始模型，一致性更好。

**跳过条件**：`global_steps > 0`（从 checkpoint 恢复）或 `len(memory_bank) > 0`（已初始化）。

### 2.5 当前默认超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `ema_alpha` | 0.9 | 等效历史窗口 ~10 步 |
| `ema_warmup_steps` | 1 | 仅跳过 count==0（EMA 未初始化）；count=0 时 query() 返回 0.0，用作 baseline 会使 A=r 虚高 |
| `mastery_threshold` | 1.0 | 8/8 全对才触发零梯度保护（7/8 对的最后一个错误 rollout 仍有有效梯度） |
| `pre_init_memory_bank` | True | 训练前 no-grad rollout 初始化 μ₀ |
| `kl_gamma` | 0.0 | 禁用（KL 惩罚方向已证伪，无法精准靶向单个 prompt） |
| `mastery_soft_alpha` | 0.0 | 禁用软性加权（已证伪：非零梯度积累导致更严重遗忘） |
| `degradation_gamma` | 0.0 | 禁用退步感知重采样（在 ≤400 题小集上无效，过拟合） |

---

## 3. 为何去掉 std 归一化

**问题根源**（已验证）：引入 EMA std 时，若第一次遇到某 prompt 时 8 rollout 全对或全错（batch_std=0），EMA std 被初始化为 1e-6，alpha=0.9 使其持续存在，warmup 结束后 `advantage = (r - μ_EMA) / 1e-6` → pg_loss 爆炸。

```
Qwen2.5-Math-1.5B 实测（cold-start v1 with EMA std）：
  step 1–260（warmup 期）：pg_loss 0.1–0.3（正常）
  step 261（warmup 结束）：pg_loss = 8004
  step 311                ：pg_loss = 20821
```

**最终方案**：彻底去掉方差 EMA，只做均值 EMA。

| 方法 | 均值 baseline | std 归一化 |
|------|--------------|-----------|
| 标准 GRPO | 当前 batch 均值 | 当前 batch std |
| Dr. GRPO | 当前 batch 均值 | ❌ 无 |
| **EMA-GRPO（当前版）** | **跨步 EMA 均值** | **❌ 无** |

---

## 4. EMA-GRPO 对截断响应的免疫机制

EMA 的 per-prompt baseline 将截断惩罚从"与整个 batch 比较"改为"与自身历史比较"：

对困难题（μ_EMA≈0.3），截断时：
$$A = 0 - 0.3 = -0.3 \quad \text{vs GRPO 的} \approx -1.5$$

惩罚强度仅为 GRPO 的 **1/5**，不产生"尽量缩短输出"的强梯度。

Mastery filter 防止简单题高 score 抬高批次均值，进一步隔离了难题的 baseline 不被简单题污染。

**这是 Qwen3-1.7B 上 EMA-GRPO AIME2024 pass@1 是 GRPO 约 6 倍的根本机制**（0.296 vs 0.050）。

---

## 5. pg_loss 符号行为：EMA-GRPO 的特殊性质

基于 Qwen2.5-Math-7B × MATH-500（225 steps）实测：

| 方法 | pg_loss 均值 | 负值步数占比 | pg_loss 范围 |
|------|------------|------------|------------|
| GRPO | 0.127 | 3.6% | [−0.018, 0.447] |
| DrGRPO | 0.056 | 6.2% | [−0.031, 0.213] |
| DAPO | 0.063 | 2.7% | [−0.018, 0.196] |
| **EMA-GRPO** | **0.035** | **28.9%** | [−0.127, 0.268] |

EMA-GRPO 的 pg_loss 符号直接反映当前 batch 相对历史基线的涨跌：

| 当前表现 | pg_loss 符号 | 含义 |
|---------|------------|------|
| 优于历史（进步） | < 0 | 正确响应贡献主导，模型被强化 |
| 持平历史（稳定） | ≈ 0 | 净更新接近零 |
| 劣于历史（退化） | > 0 | 错误响应贡献主导，模型被纠正 |

GRPO/DrGRPO/DAPO 因组内归一化使 advantage 均值为零，pg_loss 几乎恒正，无法从符号区分"进步"与"退化"。

---

## 6. 理论分析

### 方差减少

EMA 等效样本数 $n_{\text{eff}} \approx \frac{n}{1-\alpha} = \frac{8}{0.1} = 80$，均值估计方差约为批次统计的 **1/10**。

### 与标准 GRPO 的关系

- `alpha=0.0`：退化为标准 GRPO（每步 μ_new = batch_mean）
- `alpha=0.9`（默认）：等效历史窗口 ~10 步
- `alpha=0.95`：等效历史窗口 ~20 步，适合稀疏奖励（如 AIME）

### 为何 KL 惩罚方向无效（已证伪）

KL 惩罚作用于所有 token，无法只在退化题目的 token 上锚定；对某道退化题增大 KL = 对整个 batch 的 policy 施加更强约束，副作用范围远比预期宽泛。所有 KL 变体（预防式、响应式）均劣于不加 KL 的 phase3。

---

## 7. 与 Baseline 方法的对比

### vs. 标准 GRPO

| 维度 | GRPO | EMA-GRPO |
|------|------|----------|
| advantage 归一化 | 单批次（高方差） | 跨步 EMA（低方差，1/10） |
| 已掌握题目处理 | 零梯度（可被 EMA 破坏） | 零梯度（显式 mastery filter 保护） |
| 截断响应惩罚 | −1.5~−2.5（大，压缩输出） | −0.3（温和，保持长推理） |
| pg_loss 可解释性 | 几乎恒正，无方向信息 | 正负交替，直接反映进退 |

### vs. DAPO

| 维度 | DAPO | EMA-GRPO |
|------|------|----------|
| 全对 batch 处理 | 直接丢弃样本 | 零梯度但继续追踪 EMA 状态 |
| 历史信息利用 | 无（每步独立） | 跨步 EMA 积累 |
| 极高 k（pass@16）| 略低 | 更高（AIME2024 p@16 最优） |

### vs. Dr. GRPO

Dr. GRPO 去掉 std 归一化（与本方法一致），但统计量不跨步持久化，无 mastery filter。
EMA-GRPO 核心优势：**持久化的跨步 PromptMemoryBank**。

---

## 8. 已知问题：Qwen3-1.7B Divergence（高 epoch）

Qwen3-1.7B × MATH-Train-2800，step ~1390 出现思维循环（thinking loop）崩溃：

**根本原因**：高 epoch 时简单题大量触发 mastery filter，有效 batch 极小，每步梯度几乎全来自困难题的截断负信号（score=0 的长响应），最终熵爆炸（0.08→1.3），模型进入无限 "Wait, let me try again..." 循环。

**缓解方案（未验证）**：
- 限制训练 epoch ≤ 12（避开 divergence 区间）
- 增大 KL coef 0.001→0.005
- 增大 entropy_coeff 0→0.005

**最优 checkpoint 取 step 1000**（divergence 前）。

---

## 9. 实验结果汇总

### 9.1 Qwen3-1.7B × MATH-Train-2800（step 1000）

| 模型 | math500/p@1 | aime24/p@1 | aime24/p@8 | aime25/p@1 | amc/p@1 |
|------|:-----------:|:----------:|:----------:|:----------:|:-------:|
| base | 0.752 | 0.179 | 0.367 | 0.163 | 0.559 |
| grpo_step1000 | 0.691 | 0.050 | 0.167 | 0.083 | 0.369 |
| **emagrpo_step1000** | **0.802** | **0.296** | **0.467** | **0.233** | **0.656** |
| △ ema vs grpo | +0.111 | **+0.246** | **+0.300** | **+0.150** | **+0.097** |
| △ ema vs base | +0.050 | +0.117 | +0.100 | +0.070 | +0.097 |

> 路径：`/mnt/lisiqi23/grpo-forgetting-research/eval_results_mathtrain_qwen3_redo/`

GRPO 全面低于 base（Qwen3 thinking model 的截断问题最严重）；EMA-GRPO 全面超越 base，AIME2024 p@1 是 GRPO 约 **6 倍**。

### 9.2 Qwen2.5-Math-1.5B × MATH-Train-2800（step 1500 / 1720）

> 路径：`/mnt/lisiqi23/grpo-forgetting-research/eval_results_qwen25math_1.5b/`
> 评测方式：pass@1/8/16，n=16，temperature=0.6，max_tokens=4096

| 模型 | math500/p@1 | aime24/p@1 | aime24/p@8 | aime24/p@16 | aime25/p@1 | aime25/p@16 | amc/p@1 | amc/p@16 |
|------|:-----------:|:----------:|:----------:|:-----------:|:----------:|:-----------:|:-------:|:--------:|
| base | 0.553 | 0.087 | 0.225 | 0.267 | 0.033 | 0.167 | 0.402 | 0.850 |
| grpo_step1500 | 0.721 | 0.135 | 0.307 | 0.367 | 0.102 | 0.300 | 0.545 | 0.825 |
| grpo_step1720 | 0.726 | 0.113 | 0.183 | 0.200 | 0.077 | 0.267 | 0.522 | 0.850 |
| drgrpo_step1720 | 0.716 | **0.154** | 0.331 | **0.400** | 0.092 | 0.233 | 0.520 | 0.800 |
| dapo_step1500 | 0.717 | 0.142 | 0.317 | **0.400** | 0.106 | 0.300 | 0.514 | 0.875 |
| dapo_step1720 | 0.719 | 0.129 | 0.275 | 0.333 | 0.108 | 0.267 | 0.512 | **0.925** |
| emagrpo_step1500 | 0.709 | 0.119 | 0.304 | **0.400** | 0.094 | 0.300 | 0.531 | 0.800 |
| emagrpo_step1720 | 0.721 | 0.123 | 0.260 | 0.300 | **0.123** | 0.267 | 0.511 | 0.800 |
| emagrpo_coldstart_step1500 | 0.714 | **0.154** | **0.334** | 0.367 | 0.098 | 0.267 | 0.489 | 0.775 |
| **emagrpo_coldstart_step1720** | 0.723 | 0.133 | 0.342 | **0.400** | **0.123** | **0.367** | 0.528 | 0.800 |

> `emagrpo` = 旧版（含 EMA std bug）；`emagrpo_coldstart` = **当前版本**（无 std）

**关键发现**：
1. **AIME2024 p@16**：drgrpo 和 emagrpo_coldstart 并列最优（0.400），较 base +0.133
2. **AIME2025 p@16**：emagrpo_coldstart_step1720 最高（**0.367**），+0.200 vs base，所有方法最大进步
3. **GRPO step1720 AIME2024 p@16 退化**（0.267→0.200，低于 base），EMA 系列无此现象
4. 训练-测试 gap ≤ 0.01，无过拟合

### 9.3 Qwen2.5-Math-7B × MATH-500（step 75 / 150 / 225）

> 路径：`/mnt/lisiqi23/grpo-forgetting-research/eval_results_qwen25math7b_math500/`
> 评测方式：pass@1/4/8/16，n=16，temperature=0.6，max_tokens=4096

| Model | math500/p@1 | math500/p@16 | aime24/p@1 | aime24/p@8 | aime24/p@16 | aime25/p@1 | aime25/p@16 | amc/p@1 | amc/p@16 |
|-------|:-----------:|:------------:|:----------:|:----------:|:-----------:|:----------:|:-----------:|:-------:|:--------:|
| base | 0.594 | 0.888 | 0.163 | 0.384 | 0.400 | 0.075 | 0.233 | 0.473 | 0.900 |
| grpo_step75 | 0.691 | 0.896 | 0.169 | 0.425 | 0.467 | 0.083 | 0.267 | 0.558 | 0.800 |
| grpo_step150 | 0.716 | 0.886 | 0.210 | 0.475 | 0.567 | 0.108 | 0.233 | 0.563 | 0.875 |
| grpo_step225 | 0.755 | 0.904 | 0.213 | 0.445 | 0.500 | 0.090 | 0.233 | 0.592 | 0.850 |
| emagrpo_step75 | 0.692 | 0.884 | 0.194 | 0.462 | 0.533 | 0.090 | 0.300 | 0.564 | 0.875 |
| emagrpo_step150 | 0.727 | 0.902 | 0.219 | 0.431 | 0.467 | 0.092 | 0.267 | 0.584 | 0.875 |
| **emagrpo_step225** | **0.763** | 0.904 | 0.227 | 0.526 | **0.633** | 0.106 | 0.200 | 0.600 | 0.875 |
| drgrpo_step75 | 0.695 | 0.892 | 0.219 | 0.518 | 0.567 | 0.081 | 0.200 | 0.559 | 0.850 |
| drgrpo_step150 | 0.718 | 0.894 | 0.194 | 0.432 | 0.500 | 0.090 | 0.367 | 0.589 | 0.900 |
| drgrpo_step225 | 0.744 | 0.904 | 0.213 | 0.452 | 0.567 | 0.081 | 0.233 | 0.584 | 0.875 |
| dapo_step75 | 0.684 | 0.888 | 0.194 | 0.398 | 0.433 | 0.085 | 0.233 | 0.545 | 0.850 |
| dapo_step150 | 0.714 | 0.900 | 0.227 | 0.454 | 0.500 | 0.083 | 0.333 | 0.583 | **0.925** |
| **dapo_step225** | 0.757 | **0.906** | **0.258** | **0.506** | 0.567 | **0.121** | **0.433** | **0.608** | 0.875 |

**关键发现**：
1. **DAPO step225 总体最优**（p@1 指标），AIME2025 p@16=0.433 远超其他方法
2. **EMA-GRPO step225 aime2024 p@16 最高（0.633）**，高于 DAPO 和 DrGRPO（均为 0.567），极高 k 值下多样性保持优势最突出
3. 方法差异在 p@1 上较小，在 p@16（高 k）上更显著
4. 7B 未出现 Qwen3-1.7B 的熵崩塌（训练步数更少，225 steps）

### 9.4 Qwen2.5-Math-1.5B × MATH-500（step 75 / 150 / 225）

> 路径：`/mnt/lisiqi23/grpo-forgetting-research/eval_results_qwen25math1.5b_math500/`

| 模型 | math500/p@1 | aime24/p@1 | aime24/p@8 | aime24/p@16 | aime25/p@1 | amc/p@1 |
|------|:-----------:|:----------:|:----------:|:-----------:|:----------:|:-------:|
| grpo_step225 | 0.646 | 0.083 | 0.226 | 0.267 | 0.040 | 0.436 |
| **emagrpo_step75** | 0.612 | 0.104 | **0.334** | **0.433** | 0.054 | 0.463 |
| emagrpo_step225 | **0.654** | 0.100 | 0.256 | 0.333 | 0.056 | 0.453 |
| **drgrpo_step225** | 0.652 | **0.106** | **0.340** | **0.467** | **0.073** | 0.458 |
| dapo_step225 | 0.650 | 0.094 | 0.249 | 0.300 | 0.067 | 0.453 |

**关键发现**：
1. **Dr.GRPO step225 全面最优**（AIME2024 p@16=0.467 最高）
2. **EMA-GRPO 早期优势（step75）明显**：AIME2024 p@8=0.334，p@16=0.433 在 step75 全方法最高
3. GRPO AIME2024 step75→step225 明显退化（0.308→0.226），其他方法均较稳定
4. 500 题小训练集过拟合更快，最优 step 更早（75~150 vs 2800题集的 1500）

### 9.5 综合结论

| 数据集规模 | 最优方法 | EMA-GRPO 表现 |
|-----------|---------|--------------|
| 500 题小集（1.5B）| Dr.GRPO | 早期（step75）有优势，后期不稳 |
| 500 题小集（7B）| DAPO（p@1），EMA-GRPO（p@16）| 极高 k 值下 p@16 最优 |
| 2800 题大集（1.5B）| 接近 DrGRPO，AIME2025 p@16 最优 | AIME2025 p@16 所有方法最大提升 |
| 2800 题大集（1.7B）| EMA-GRPO（GRPO 不可用）| AIME 提升 6 倍，全面超越 base |

---

## 10. 前置条件与使用方式

### 数据集要求

parquet 文件的 `extra_info` 必须包含 `index`（整数，数据集行号）：
```python
extra_info = {"index": 0, "split": "train"}
```

### 标准启动命令

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ema_grpo \
    algorithm.ema_alpha=0.9 \
    algorithm.ema_warmup_steps=1 \
    algorithm.mastery_threshold=1.0 \
    algorithm.pre_init_memory_bank=true \
    algorithm.kl_gamma=0.0 \
    algorithm.use_kl_in_reward=False \
    ...
```

### alpha 消融参数

```bash
algorithm.ema_alpha=0.0    # 等价于标准 GRPO
algorithm.ema_alpha=0.5    # 等效历史窗口 ~2 步
algorithm.ema_alpha=0.7    # 等效历史窗口 ~3 步
algorithm.ema_alpha=0.9    # 默认，~10 步
algorithm.ema_alpha=0.95   # ~20 步，适合稀疏奖励（AIME）
```

### LoRA 合并注意事项

`legacy_model_merger.py` 硬编码 `lora_alpha=0`，必须使用修复后的 `convert_checkpoint.py`：
```bash
python3 tools/convert_checkpoint.py \
    --checkpoint_dir .../actor \
    --output_dir .../hf \
    --merge_lora --lora_alpha 32
```
