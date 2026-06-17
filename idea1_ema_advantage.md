# 想法一：跨 Epoch EMA 统计的 Advantage 归一化

## 1. 问题分析

### 1.1 标准 GRPO 的 Advantage 计算

标准 GRPO 对每个 prompt $p$ 采样 $n$ 个响应，计算 advantage：

$$A_i = \frac{r_i - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \varepsilon}, \quad \mu_{\text{batch}} = \frac{1}{n}\sum_{j=1}^n r_j, \quad \sigma_{\text{batch}} = \sqrt{\frac{1}{n}\sum_{j=1}^n (r_j - \mu_{\text{batch}})^2}$$

这里的均值和方差**仅由当前批次的 n 个 rollout 估计**。

### 1.2 为什么这是不稳定的

对于 0/1 奖励的数学题，假设 prompt $p$ 的真实答对概率为 $\pi(p)$，则：

- $\mu_{\text{true}}(p) = \pi(p)$
- $\sigma_{\text{true}}(p) = \sqrt{\pi(p)(1-\pi(p))}$

用 $n=8$ 个 rollout 估计时，$\mu_{\text{batch}}$ 的标准误差为：

$$\text{SE}(\mu_{\text{batch}}) = \frac{\sigma_{\text{true}}}{\sqrt{n}} = \frac{\sqrt{\pi(p)(1-\pi(p))}}{\sqrt{8}}$$

**当 $\pi(p) = 0.5$（中等难度题）时，SE ≈ 0.177**，即均值估计的误差高达 ±0.177。

这意味着：
- 同一道题在不同 step 中，advantage 的基准线可以相差很大
- "幸运批次"（8个都答对）和"倒霉批次"（8个都答错）产生截然不同的梯度方向
- 模型在同一道题上收到矛盾的训练信号，导致策略震荡

**标准 GRPO 的隐式保护机制**：当 $n$ 个 rollout **全部答对**时，
$\mu_{\text{batch}} = 1, \sigma_{\text{batch}} = 0$，所有 advantage 自动为 0，
产生零梯度，模型不会被强化"已经完全掌握"的题目。
**EMA-GRPO 破坏了这一机制**：EMA 均值滞后（如 $\hat{\mu}=0.7$），
导致全对 rollout 的 advantage 为正，模型在已掌握题目上浪费梯度，
挤占对未掌握题目的学习资源，甚至因梯度干扰引发遗忘。

### 1.3 实验中的直接证据

对两组实验（EMA-GRPO alpha=0.9 与标准 GRPO）的 rollout_logs 分析显示：

- EMA-GRPO 退化题数：37 道；标准 GRPO 退化题数：19 道
- 退化题两组重叠仅 6 道，说明 EMA 引入了额外的退化
- 标准 GRPO 中退化题大量属于两类：
  - **Type A（高波动）**：全程 acc 震荡（vol > 0.14），从未稳定掌握，endpoint 碰巧落在低位
  - **Type B（近完美漂移）**：连续 7-8 步 acc=1.0，最后一步偶尔 0.875，实为采样方差而非遗忘
- 用线性斜率 + 前后半段均值三重验证，真正存在系统性退化的题仅约 14 道

**结论**：遗忘的根本原因不是归一化噪声，而是 EMA 均值滞后导致已掌握题目被错误强化，
以及部分题目的策略本身高波动（policy instability），这与残差连接的思想高度吻合——
训练应在探索新能力的同时，通过跳跃连接保护已掌握的能力。

---

## 2. 方法设计

### 2.1 核心思想

用**指数移动平均（EMA）**跨 epoch 积累每个 prompt 的历史统计，
用更稳定的历史统计替代噪声大的批次统计来归一化 advantage。

### 2.2 PromptMemoryBank

为每个 prompt 维护以下状态：

| 字段 | 符号 | 含义 |
|------|------|------|
| EMA 均值 | $\hat{\mu}_t(p)$ | prompt $p$ 在第 $t$ 次出现时的历史平均得分 |
| EMA 方差 | $\hat{\sigma}^2_t(p)$ | prompt $p$ 的历史得分方差 |
| 出现次数 | $k(p)$ | 用于 warm-up 判断 |

**EMA 更新规则**（每次该 prompt 出现时更新）：

$$\hat{\mu}_t(p) = \alpha \cdot \hat{\mu}_{t-1}(p) + (1 - \alpha) \cdot \mu_{\text{batch}}(p)$$

$$\hat{\sigma}^2_t(p) = \alpha \cdot \hat{\sigma}^2_{t-1}(p) + (1 - \alpha) \cdot \left[\sigma^2_{\text{batch}}(p) + (\mu_{\text{batch}}(p) - \hat{\mu}_t(p))^2\right]$$

注：方差更新式中的第二项用于捕捉批次均值与历史均值之间的偏差，避免低估方差。

**初始化**：
$$\hat{\mu}_0(p) = 0.0, \quad \hat{\sigma}^2_0(p) = 1.0, \quad k(p) = 0$$

### 2.3 带 Warm-up 的 Advantage 归一化

$$A_i^{\text{EMA}} = \begin{cases}
\dfrac{r_i - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \varepsilon} & \text{if } k(p) < k_{\text{warmup}} \\[10pt]
\dfrac{r_i - \hat{\mu}(p)}{\hat{\sigma}(p) + \varepsilon} & \text{if } k(p) \geq k_{\text{warmup}}
\end{cases}$$

其中 $k_{\text{warmup}}$ 为 warm-up 阈值（建议 3~5 次出现，即约 3~5 个 epoch 后生效）。

### 2.4 全对样本过滤 + 掌握度自适应 KL 增强

**核心思想（残差连接类比）**：

```
标准训练流:  所有 prompt → GRPO 梯度更新
残差思想:    已掌握 prompt → 跳过梯度主路径（零 advantage）
                           + 增强 KL 锚定（跳跃连接，防止 policy 漂移）
             未掌握 prompt → 正常 EMA-GRPO 梯度更新（主路径探索）
```

**全对样本过滤**：

当 prompt $p$ 的当前批次全部答对（$\text{acc\_batch}(p) = 1.0$）时，
无论 EMA 均值是否滞后，**强制将该 prompt 的所有 advantage 置零**：

$$A_i^{\text{final}} = \begin{cases}
0 & \text{if } \text{acc\_batch}(p) = 1.0 \quad \text{（全对过滤）} \\
A_i^{\text{EMA}} & \text{otherwise}
\end{cases}$$

这恢复了标准 GRPO 的隐式保护机制，避免 EMA 滞后引发的虚假正 advantage。

**掌握度自适应 KL 增强**：

同时，对全对 prompt 施加增强的 KL 惩罚，防止 policy 在这些 token 上随其他梯度漂移：

$$\beta_{\text{eff}}(p) = \beta_{\text{base}} \cdot \left(1 + \gamma_{\text{kl}} \cdot \hat{\mu}(p)\right)$$

其中 $\hat{\mu}(p)$ 为 EMA 历史均值（反映长期掌握程度），$\gamma_{\text{kl}}$ 为放大系数（建议 1.0~3.0）。

- 历史平均答对率高的题（$\hat{\mu} \approx 1.0$）：KL 系数最大，policy 锚定最紧（跳跃连接最强）
- 历史平均答对率低的题（$\hat{\mu} \approx 0$）：KL 系数接近 baseline，允许自由探索

**扩展：软过滤（非全对时的平滑版本）**：

全对过滤是硬阈值，可推广为基于历史掌握度的软权重：

$$w(p) = 1 - \hat{\mu}(p)^{\lambda}$$

其中 $\lambda > 0$ 控制软化程度（$\lambda=1$ 为线性，$\lambda \to \infty$ 退化为硬过滤）。
advantage 乘以 $w(p)$，历史答对率越高，梯度贡献越小。

### 2.5 与标准 GRPO 的对比

| | 标准 GRPO | EMA-GRPO（原版） | EMA-GRPO + 全对过滤 + KL增强 |
|--|---------|-----------------|----------------------------|
| 均值来源 | batch | EMA 历史 | EMA 历史 |
| 方差来源 | batch | EMA 历史 | EMA 历史 |
| 全对零梯度 | 自动（σ=0） | **破坏**（EMA 滞后） | **恢复**（显式过滤） |
| 已掌握题 KL | 固定 β | 固定 β | **增强**（自适应 β） |
| 残差保护 | 无 | 无 | **有**（过滤 + KL 双重保护） |
| 额外存储 | 无 | O(N) 浮点 | O(N) 浮点 |

---

## 3. 算法流程

```
训练循环每步：
  输入：batch 中的 prompts P = {p_1, ..., p_B}

  Step 1: vLLM Rollout
    对每个 p_i 采样 n 个响应，得到 {r_{i,1}, ..., r_{i,n}}

  Step 2: 计算批次统计
    μ_batch(p_i) = mean({r_{i,j}})
    σ_batch(p_i) = std({r_{i,j}})

  Step 3: 查询 PromptMemoryBank
    对每个 p_i：
      key_i = hash(p_i)
      读取 (μ̂, σ̂, k) = MemoryBank[key_i]

  Step 4: 计算 EMA-Advantage（含全对过滤）
    对每个 (p_i, r_{i,j})：
      if μ_batch(p_i) == 1.0:          # 全对过滤：恢复零梯度保护
        A_{i,j} = 0.0
      elif k_i < k_warmup:
        A_{i,j} = (r_{i,j} - μ_batch(p_i)) / (σ_batch(p_i) + ε)   # 标准 GRPO
      else:
        A_{i,j} = (r_{i,j} - μ̂_i) / (σ̂_i + ε)                    # EMA-GRPO

  Step 5: 计算掌握度自适应 KL 系数
    对每个 p_i：
      β_eff(p_i) = β_base * (1 + γ_kl * μ̂_i)
      # μ̂_i ≈ 1.0 → KL 系数最大，锚定 policy（跳跃连接）
      # μ̂_i ≈ 0.0 → KL 系数接近 baseline，允许自由探索
      将 β_eff(p_i) 作为该 prompt 对应 token 的 per-token KL 权重

  Step 6: 更新 MemoryBank（在 advantage 计算之后更新，避免用未来信息）
    对每个 p_i：
      μ̂_new = α * μ̂ + (1-α) * μ_batch(p_i)
      σ̂²_new = α * σ̂² + (1-α) * [σ²_batch(p_i) + (μ_batch(p_i) - μ̂_new)²]
      k_new = k + 1
      MemoryBank[key_i] = (μ̂_new, √σ̂²_new, k_new)

  Step 7: 执行 PPO 更新（使用 EMA-Advantage + 自适应 KL 权重）
```

---

## 4. 实现细节

### 4.1 PromptMemoryBank 设计

```python
# verl/utils/prompt_memory_bank.py

class PromptMemoryBank:
    """
    跨训练步维护 per-prompt 的 EMA 统计量。
    支持多进程安全访问（Ray Actor 模式）。
    """

    def __init__(self, alpha=0.9, warmup_steps=3, eps=1e-8,
                 mastery_threshold=1.0, kl_gamma=2.0):
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.mastery_threshold = mastery_threshold  # 全对过滤阈值，默认 1.0
        self.kl_gamma = kl_gamma                    # KL 增强系数 γ_kl
        # key: prompt_hash -> (ema_mean, ema_std, count)
        self.bank: dict[str, tuple[float, float, int]] = {}

    def _hash(self, prompt: str) -> str:
        import hashlib
        return hashlib.md5(prompt[:500].encode()).hexdigest()

    def query(self, prompt: str) -> tuple[float, float, int]:
        """返回 (ema_mean, ema_std, count)"""
        key = self._hash(prompt)
        return self.bank.get(key, (0.0, 1.0, 0))

    def update(self, prompt: str, batch_mean: float, batch_std: float):
        """用本批次统计更新 EMA"""
        key = self._hash(prompt)
        mu_old, std_old, k = self.bank.get(key, (0.0, 1.0, 0))

        mu_new = self.alpha * mu_old + (1 - self.alpha) * batch_mean
        var_old = std_old ** 2
        var_new = self.alpha * var_old + (1 - self.alpha) * (
            batch_std ** 2 + (batch_mean - mu_new) ** 2
        )
        std_new = max(var_new ** 0.5, self.eps)
        self.bank[key] = (mu_new, std_new, k + 1)

    def is_warmed_up(self, prompt: str) -> bool:
        _, _, k = self.query(prompt)
        return k >= self.warmup_steps

    def is_mastered_batch(self, batch_mean: float) -> bool:
        """判断当前 batch 是否全对（触发过滤）"""
        return batch_mean >= self.mastery_threshold

    def kl_multiplier(self, prompt: str) -> float:
        """返回该 prompt 的 KL 系数倍率：1 + γ_kl * ema_mean"""
        mu_hat, _, _ = self.query(prompt)
        return 1.0 + self.kl_gamma * mu_hat

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.bank, f)

    def load(self, path: str):
        import json
        with open(path) as f:
            self.bank = json.load(f)
```

### 4.2 与 verl 的集成点

需要修改的文件：`verl/trainer/ppo/ray_trainer.py`

在 `_update_actor` 调用前，拦截 advantage 计算过程：

```python
# 在 ray_trainer.py 中：
# 1. 初始化 MemoryBank（__init__ 中）
self.prompt_memory = PromptMemoryBank(
    alpha=config.algorithm.get("ema_alpha", 0.9),
    warmup_steps=config.algorithm.get("ema_warmup_steps", 3),
    mastery_threshold=config.algorithm.get("mastery_threshold", 1.0),
    kl_gamma=config.algorithm.get("kl_gamma", 2.0),
)

# 2. 在 rollout 之后、actor update 之前，修改 advantage 和 KL 权重
# 伪代码，具体位置在 fit() 的主循环中
kl_weights = torch.ones_like(batch.advantages)  # 默认 KL 权重 = 1.0

for i, prompt in enumerate(batch_prompts):
    mu_hat, std_hat, k = self.prompt_memory.query(prompt)
    mask = (prompt_index == i)

    if self.prompt_memory.is_mastered_batch(batch_mean[i]):
        # 全对过滤：零梯度 + 增强 KL（残差跳跃连接）
        batch.advantages[mask] = 0.0
        kl_weights[mask] = self.prompt_memory.kl_multiplier(prompt)
    elif k >= warmup_steps:
        # EMA-GRPO：用历史统计归一化
        batch.advantages[mask] = (batch.rewards[mask] - mu_hat) / (std_hat + eps)
    # else: 保持标准 GRPO advantage（warmup 阶段）

    # 更新 MemoryBank（advantage 计算后更新，避免用未来信息）
    self.prompt_memory.update(prompt, batch_mean[i], batch_std[i])

# 3. 将 kl_weights 传入 PPO 更新，作用于 per-token KL loss
# 具体实现依赖 verl actor 的 loss 接口，需在 compute_policy_loss 中支持
# kl_loss = (kl_weights * kl_per_token).mean()

# 4. 每隔 N 步保存 MemoryBank（checkpoint 一起保存）
self.prompt_memory.save(os.path.join(checkpoint_dir, "prompt_memory.json"))
```

### 4.3 Checkpoint 集成

MemoryBank 需要随模型一起保存和恢复，否则重启训练时历史统计丢失：
- 保存：在 `trainer.save_freq` 触发时，随 checkpoint 一起保存 `prompt_memory.json`
- 恢复：在 `trainer.resume_from` 时，同时加载 `prompt_memory.json`

---

## 5. 超参数分析

### 5.0 全对过滤阈值与 KL 增强系数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `mastery_threshold` | `1.0` | 全部 n 个 rollout 答对才触发过滤，严格定义 mastery |
| `kl_gamma` | `1.0~3.0` | KL 倍率放大系数；=0 退化为无 KL 增强；过大可能导致学习停滞 |

**为什么用 `mastery_threshold=1.0` 而非 0.875**：
- 0.875 意味着 8 个 rollout 中 7 个对，但 1 个错的 rollout 仍有有效梯度信号
- 全对（1.0）时梯度信号才真正为零，此时过滤纯粹是为了修复 EMA 滞后的虚假梯度

**`kl_gamma` 的选择**：
- `kl_gamma=1.0`：`β_eff` 最大为 `2β`，保守
- `kl_gamma=2.0`：`β_eff` 最大为 `3β`，推荐起点
- `kl_gamma=4.0`：`β_eff` 最大为 `5β`，强锚定，适合严重遗忘的场景

### 5.1 EMA 衰减系数 α

| α | 等效历史窗口长度 | 特性 |
|---|----------------|------|
| 0.99 | ~100 步 | 非常稳定，对近期变化不敏感，早期统计影响久 |
| 0.95 | ~20 步 | **推荐**，约对应 1 个 epoch（25步）的平滑 |
| 0.90 | ~10 步 | 对近期变化较敏感，适合小数据集 |
| 0.80 | ~5 步 | 接近批次统计，稳定性提升有限 |

等效窗口长度计算：$L \approx \frac{1}{1-\alpha}$

对本实验（每 epoch 25 步，每道题约每 epoch 出现 1 次）：
- **推荐 α = 0.9**：等效约 10 步历史，平衡稳定性和对学习进度的跟踪

### 5.2 Warm-up 阈值 k_warmup

| k_warmup | 含义 | 风险 |
|----------|------|------|
| 1 | 第一次出现就用 EMA | EMA 初始值 (0.0, 1.0) 可能严重偏差 |
| 3 | 约 3 个 epoch 后生效 | **推荐**，有足够历史 |
| 5 | 约 5 个 epoch 后生效 | 保守，前半段训练没有改善 |

### 5.3 ε（防除零项）

建议 `ε = 1e-6`。对于 0/1 奖励，当 prompt 总是被答对或总是答错时，
`std → 0`，`ε` 防止 advantage 爆炸。

---

## 6. 理论分析

### 6.1 方差减少量

设 prompt $p$ 历史出现了 $k$ 次，每次 $n$ 个 rollout，真实概率为 $\pi$。

- 标准 GRPO 的均值估计方差：$\text{Var}(\mu_{\text{batch}}) = \frac{\pi(1-\pi)}{n}$
- EMA 均值的有效等效样本数：$n_{\text{eff}} \approx \frac{n}{1-\alpha} = \frac{8}{0.1} = 80$（α=0.9时）

**方差减少比**：$\frac{\text{Var}(\hat{\mu}_{\text{EMA}})}{\text{Var}(\mu_{\text{batch}})} \approx \frac{1}{(1-\alpha)^{-1}} = 1-\alpha = 0.1$

即 EMA 方法在稳定后，均值估计方差约为标准 GRPO 的 **1/10**。

### 6.2 与 GRPO 的关系

EMA-GRPO 可以看作标准 GRPO 的一个特例：
- 当 α=0（无历史记忆）时，退化为标准 GRPO
- 当 α→1（完全历史记忆）时，每道题有固定的归一化基准，equivalent to per-prompt whitening

### 6.3 潜在的负面效应

如果模型确实在某道题上**合理地进步了**（从 mean=0.5 提升到 mean=1.0），
EMA 均值的滞后性会导致 advantage 的正值被低估（因为 EMA_mean 还停留在 0.5 附近）。
这可能轻微减慢学习速度，但不会阻止学习方向。

---

## 7. 对照实验设计

### 7.1 实验设置

| 配置 | 说明 |
|------|------|
| **Baseline** | 标准 GRPO，当前脚本 |
| **EMA-GRPO (α=0.9)** | 原版 EMA，α=0.9，k_warmup=3，无过滤无 KL 增强 |
| **EMA-GRPO + filter** | 原版 EMA + 全对过滤，无 KL 增强（kl_gamma=0） |
| **EMA-GRPO + filter + KL (γ=2)** | 全对过滤 + KL 增强，γ=2.0（推荐主实验） |
| **EMA-GRPO + filter + KL (γ=4)** | 消融：更强的 KL 锚定 |

消融顺序建议：先验证 `filter` 单独效果（应接近 baseline），再叠加 KL 增强，
以分离过滤和 KL 增强各自的贡献。

所有其他超参数完全相同，数据集、模型、random seed 一致。

### 7.2 评估指标（除标准 acc 之外）

1. **退步题数量**：每个 epoch 结束后，统计从 epoch 1 基准得分下降 > 0.25 的题目数
2. **得分标准差**：每道题跨 epoch 得分的标准差，越低越稳定
3. **Advantage 方差**：训练过程中 advantage 值的方差，反映梯度稳定性
4. **难题恢复速度**：退步后重新答对所需的 epoch 数

### 7.3 假设与预期

- **H1**：EMA-GRPO（原版）退步题多于 baseline ✗（已被实验否定，37 vs 19）
- **H2**：EMA-GRPO + 全对过滤 退步题接近 baseline（修复 EMA 破坏的零梯度保护）
- **H3**：EMA-GRPO + 过滤 + KL增强 退步题少于 baseline（KL 增强提供额外锚定）
- **H4**：KL 增强不显著损害整体准确率（已掌握题不参与梯度，只参与 KL 约束）
- **H5**：γ=2 优于 γ=4（过强锚定可能阻碍模型在已掌握题上的表达调整）

---

## 8. 与现有工作的联系

| 方法 | 相似点 | 差异 |
|------|--------|------|
| **DAPO** (Yu et al., 2024) | 去掉 std 归一化，减少噪声 | 本方法保留归一化但用更稳定的历史统计 |
| **Dr. GRPO** (Liu et al., 2025) | 识别 GRPO 归一化的偏差问题 | 本方法从跨步稳定性角度入手，而非单步偏差 |
| **Running statistics in PPO** | 在线更新均值/方差 | 本方法是 per-prompt 的，非全局 |
| **Prioritized Experience Replay** | 维护样本级历史 | 本方法用于归一化而非采样优先级 |

---

## 9. 文件结构

```
grpo-forgetting-research/
├── idea1_ema_advantage.md          # 本文件
├── src/
│   ├── prompt_memory_bank.py       # PromptMemoryBank 实现
│   └── ema_grpo_trainer.py         # 继承 RayPPOTrainer 的修改版
├── experiments/
│   ├── run_baseline.sh             # 标准 GRPO 基准
│   ├── run_ema_alpha09.sh          # EMA α=0.9
│   └── run_ema_alpha095.sh         # EMA α=0.95
└── analysis/
    └── compare_forgetting.py       # 对比退步题数量的分析脚本
```

---

## 10. 实现 Checklist

- [ ] 实现 `PromptMemoryBank`（带 save/load，新增 `mastery_threshold`、`kl_gamma`、`kl_multiplier`）
- [ ] 在 `ray_trainer.py` 中找到 advantage 计算的位置
- [ ] 在 rollout 之后插入 EMA-advantage 替换逻辑
- [ ] 插入全对过滤逻辑：`acc_batch == 1.0` 时强制 `advantage = 0`
- [ ] 构造 per-token `kl_weights` tensor，对全对 prompt 的 token 赋值 `kl_multiplier(prompt)`
- [ ] 在 PPO loss 计算中支持 `kl_weights`（修改 `compute_policy_loss` 或等价接口）
- [ ] 在 checkpoint 保存/加载中集成 MemoryBank
- [ ] 添加 hydra config 项：`algorithm.ema_alpha`、`algorithm.ema_warmup_steps`、`algorithm.mastery_threshold`、`algorithm.kl_gamma`
- [ ] 编写消融实验脚本（filter-only / filter+KL_γ2 / filter+KL_γ4）
- [ ] 验证：全对 batch 时 advantage 确为 0，kl_weights 确为 `1 + γ * μ̂`
- [ ] 验证：warm-up 阶段输出与标准 GRPO 一致
- [ ] 验证：EMA 更新在 step 1 后值合理
