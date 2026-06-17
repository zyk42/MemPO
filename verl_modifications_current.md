# verl 源码改动说明：EMA-GRPO 当前版本

## 概述

本文档描述为支持 **EMA-GRPO**（`algorithm.adv_estimator=ema_grpo`）对 verl 源码所做的改动（当前最终版本）。

**核心思想**：用每个 prompt 跨训练步积累的 EMA 均值作为 advantage baseline，不使用 std 归一化：

$$A_i^{(p)} = r_i^{(p)} - \mu_{\text{EMA}}(p)$$

**自适应 α（无参数，由 rollout n 完全决定）**：

$$\alpha_t(p) = \frac{1}{n} + \frac{n-2}{n} \cdot \left(1 - \left|\hat{\mu}_t(p) - \mu_{t-1}(p)\right|\right)$$

- 当前批次均值 $\hat{\mu}_t$ 与历史均值 $\mu_{t-1}$ 偏差大（策略快速变化）→ $\alpha \to 1/n$，信任当前批次，减少基线滞后
- 偏差小（策略稳定）→ $\alpha \to (n-1)/n$，保持历史平滑
- 对 n=8：$\alpha \in [0.125,\ 0.875]$，无需手动调参

**掌握度过滤**：当**更新后的 EMA 均值** $\mu_t$（不是当前批次均值 $\hat{\mu}_t$）严格超过阈值 $\tau = (n-1)/n$ 时，强制 advantage = 0（零梯度保护）。用 $\mu_t$ 而非 $\hat{\mu}_t$ 的好处：单次幸运全对批次不会误触发过滤，只有模型在多步中持续答对（$\mu_t$ 积累到 $> \tau$）才触发。

**当前默认超参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout_n` | 8（来自 `actor_rollout_ref.rollout.n`） | 决定动态 α 范围和 τ |
| `ema_warmup_steps` | 1 | `pre_init=False` 时跳过未初始化的 count=0 情况 |
| `mastery_threshold` | **(n-1)/n（自动推导）** | 不再是固定值 |
| `pre_init_memory_bank` | True | 训练前 no-grad rollout 初始化 μ₀ |
| `kl_gamma` | 0.0（禁用） | |
| `mastery_soft_alpha` | 0.0（禁用） | |
| `degradation_gamma` | 0.0（禁用） | |

> `ema_alpha`（旧参数）**已废弃**，改用无参数动态 α（见下文），保留字段仅为向后兼容。

---

## 改动文件清单

### 1. `verl/utils/prompt_memory_bank.py` *(新增文件)*

核心新组件。以数据集行号（`dataset_index`，跨 epoch 稳定）为键，存储每个 prompt 的 `(ema_mean, count)` 二元组。

**关键设计：**
- `update()` 在 `count==0` 时 cold-start 直接初始化（`mu_new = batch_mean`），不从 0.0 起步
- `query()` 对未见过的 prompt 返回 `(0.0, 1.0, 0)`（count=0，未初始化）
- `update()` 必须在 `compute_advantage()` **之后**调用，防止未来信息泄露
- 不追踪 EMA std（去除方差估计偏差），`query()` 的 std 位置始终返回 dummy 值 1.0
- **动态 α**（无参数）：`α_t(p) = 1/n + (n-2)/n · (1 - |μ̂_t - μ_{t-1}|)`
- **掌握度阈值** τ = (n-1)/n，过滤条件为 `batch_mean > τ`（严格大于，即全部 n 个 rollout 正确）

**动态 α 的直觉：**

| 情形 | δ = |μ̂_t - μ_{t-1}| | 计算得 α | 含义 |
|------|-----------------|---------|------|
| 策略大幅改进（μ 跳跃） | δ → 1 | α → 1/n | 基线滞后严重，信任当前批次 |
| 策略稳定（μ 不变） | δ → 0 | α → (n-1)/n | 基线准确，保持历史平滑 |

对 n=8：α ∈ [0.125, 0.875]，无需手动调参。

**关键方法：**

```python
def query(self, dataset_index) -> tuple[float, float, int]:
    entry = self._bank.get(self._key(dataset_index))
    if entry is None:
        return (0.0, 1.0, 0)
    ema_mean, count = entry
    return (ema_mean, 1.0, count)

def update(self, dataset_index, batch_mean, batch_std):
    key = self._key(dataset_index)
    entry = self._bank.get(key)
    mu_old, count = entry if entry is not None else (0.0, 0)
    if count == 0:
        mu_new = batch_mean          # cold-start：直接初始化
        if key not in self._initial_acc:
            self._initial_acc[key] = batch_mean
    else:
        # 无参数动态 alpha
        n = self.rollout_n
        delta = abs(batch_mean - mu_old)
        alpha = 1.0 / n + ((n - 2) / n) * (1.0 - delta)
        alpha = max(1.0 / n, min((n - 1) / n, alpha))
        mu_new = alpha * mu_old + (1.0 - alpha) * batch_mean
    self._bank[key] = (mu_new, count + 1)

def is_mastered_batch(self, batch_mean) -> bool:
    # τ = (n-1)/n，严格大于才触发（即全部 n 个 rollout 正确时 batch_mean=1.0 > τ）
    return batch_mean > self.mastery_threshold
```

**构造参数：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `rollout_n` | 8 | rollout 数量；决定 α ∈ [1/n, (n-1)/n] 和 τ = (n-1)/n |
| `warmup_steps` | 1 | **当 `pre_init_memory_bank=True` 时此参数在正常流程中不生效**（pre-init 后所有 prompt count=1，条件 `count < 1` 永远为假）。仅作为 `pre_init=False` 时的保险丝：count=0 时 query() 返回 0.0（未初始化），若不跳过会使 advantage=r 虚高 |
| `kl_gamma` | 0.0 | KL 倍率系数（设 0 禁用） |
| `mastery_soft_alpha` | 0.0 | 软性加权指数（设 0 禁用，使用硬截断过滤） |
| `eps` | 1e-6 | 防除零下界（当前无 std 除法，预留） |

**持久化：** `save()` / `load()` 使用 JSON，同步持久化 `_initial_acc`；`load()` 兼容旧格式（三元组 `(mean, std, count)`）：
```python
bank[k] = (v[0], v[-1])  # 取首位（mean）和末位（count），两种格式均适用
```

---

### 2. `verl/trainer/ppo/core_algos.py`

**改动内容：**
- `AdvantageEstimator` 枚举新增 `EMA_GRPO = "ema_grpo"`
- 新增 `compute_grpo_ema_outcome_advantage()` 函数

**函数签名：**
```python
@register_adv_est(AdvantageEstimator.EMA_GRPO)
def compute_grpo_ema_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,          # 当前步内批次分组 uid（同标准 GRPO）
    prompt_index: np.ndarray,   # 跨 epoch 稳定的数据集行号
    memory_bank: PromptMemoryBank,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**计算逻辑：**
1. 按 `uid` 分组，计算每个 prompt 的**当前批次均值** $\hat{\mu}_t$（`batch_mean`）
2. 先调用 `memory_bank.update()` 得到**更新后的 EMA 均值** $\mu_t = \alpha_t \mu_{t-1} + (1-\alpha_t)\hat{\mu}_t$
3. 再 `query()` 读取 $\mu_t$，用于后续 mastery 判断和 baseline
4. Baseline 选择逻辑：
   - 若 `is_mastered_batch(μ_t)`（即 $\mu_t > \tau$，用**更新后 EMA 均值**判断）：baseline = $\hat{\mu}_t$ → advantage = r − 1 = 0（持续掌握确认，零梯度保护）
   - 否则：baseline = $\mu_t$（**更新后 EMA 均值**）
5. `advantage = score - baseline`（不除 std）

---

### 2.5 `verl/experimental/dataset/degradation_sampler.py` *(新增文件，可选)*

实现 `DegradationAwareSampler`，继承 `AbstractCurriculumSampler`。
对退步题目进行过采样：`weight[p] = 1 + gamma * (difficulty_score(p) + degradation_score(p))`

**仅在 `degradation_gamma > 0` 时使用，默认禁用。需 `data.dataloader_num_workers=0`。**

---

### 3. `verl/trainer/ppo/ray_trainer.py`

#### 3.1 import
```python
from verl.utils.prompt_memory_bank import PromptMemoryBank
```

#### 3.2 `__init__()` — PromptMemoryBank 初始化

在 `_create_dataloader()` 调用**之前**初始化（`DegradationAwareSampler` 在 `_create_dataloader` 内部构造时需要 memory bank 已存在）。
`rollout_n` 从 `config.actor_rollout_ref.rollout.n` 自动读取，不再需要 `ema_alpha` 或 `mastery_threshold`：

```python
if config.algorithm.adv_estimator == AdvantageEstimator.EMA_GRPO:
    rollout_n = config.actor_rollout_ref.rollout.n
    self.prompt_memory_bank = PromptMemoryBank(
        rollout_n=rollout_n,
        warmup_steps=config.algorithm.get("ema_warmup_steps", 1),
        kl_gamma=config.algorithm.get("kl_gamma", 0.0),
        mastery_soft_alpha=config.algorithm.get("mastery_soft_alpha", 0.0),
    )
else:
    self.prompt_memory_bank = None

self._create_dataloader(...)
```

#### 3.3 `_create_dataloader()` — 保存 collate_fn 并可选创建 DegradationAwareSampler

在 default collate_fn 赋值之后保存：
```python
self.collate_fn = collate_fn
```

可选的 DegradationAwareSampler（仅 `degradation_gamma > 0` 时生效）：
```python
if train_sampler is None:
    degradation_gamma = self.config.algorithm.get("degradation_gamma", 0.0)
    if (
        self.config.algorithm.adv_estimator == AdvantageEstimator.EMA_GRPO
        and degradation_gamma > 0.0
        and getattr(self, "prompt_memory_bank", None) is not None
    ):
        from verl.experimental.dataset.degradation_sampler import DegradationAwareSampler
        train_sampler = DegradationAwareSampler(
            data_source=self.train_dataset,
            data_config=self.config.data,
            memory_bank=self.prompt_memory_bank,
            gamma=degradation_gamma,
        )
    else:
        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
```

#### 3.4 `compute_advantage()` — EMA_GRPO 分支

函数签名新增 `memory_bank: Optional[PromptMemoryBank] = None`；EMA_GRPO 分支：
```python
elif adv_estimator == AdvantageEstimator.EMA_GRPO:
    assert memory_bank is not None
    if "index" in data.non_tensor_batch:
        prompt_index = data.non_tensor_batch["index"]
    else:
        prompt_index = np.array([ei["index"] for ei in data.non_tensor_batch["extra_info"]])
    advantages, returns = core_algos.compute_grpo_ema_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        prompt_index=prompt_index,
        memory_bank=memory_bank,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

#### 3.5 `apply_kl_penalty()` — 掌握度自适应 KL 增强（可选，默认禁用）

函数签名新增 `memory_bank=None`。当 `memory_bank.kl_gamma > 0` 时构造 per-sample KL 倍率并应用：
```python
kl_mult[i, 0] = memory_bank.kl_multiplier(prompt_index[i])
token_level_rewards = token_level_scores - beta * kl_mult * kld
```
`memory_bank=None` 或 `kl_gamma=0` 时行为与原版完全一致。

#### 3.6 `fit()` — 传递 memory_bank

```python
batch, kl_metrics = apply_kl_penalty(
    batch, kl_ctrl=self.kl_ctrl_in_reward,
    kl_penalty=self.config.algorithm.kl_penalty,
    memory_bank=self.prompt_memory_bank,
)
batch = compute_advantage(batch, ..., memory_bank=self.prompt_memory_bank)
```

#### 3.7 `fit()` — pre_init_memory_bank 调用

在 `val_before_train` 之后、训练循环之前：
```python
if (
    self.config.algorithm.adv_estimator == AdvantageEstimator.EMA_GRPO
    and self.config.algorithm.get("pre_init_memory_bank", True)
    and self.prompt_memory_bank is not None
    and self.global_steps == 0
    and len(self.prompt_memory_bank) == 0
):
    self._pre_init_memory_bank()
```

#### 3.8 `_pre_init_memory_bank()` — 新增方法

训练前对全体训练 prompt 运行一次 no-grad rollout pass，将每道题的 batch_mean 直接写入 memory bank（cold-start 初始化 μ₀）：

```python
def _pre_init_memory_bank(self):
    from torch.utils.data import DataLoader
    init_dataloader = DataLoader(
        dataset=self.train_dataset,
        batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
        num_workers=0, drop_last=False, collate_fn=self.collate_fn,
    )
    for batch_dict in init_dataloader:
        batch = DataProto.from_single_dict(batch_dict)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        gen_batch = self._get_gen_batch(batch)
        gen_batch_output = gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
        )
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
        self.checkpoint_manager.sleep_replicas()
        gen_batch_output.meta_info.pop("timing", None)
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        reward_tensor, _ = extract_reward(batch)
        batch.batch["token_level_rewards"] = reward_tensor
        if "index" in batch.non_tensor_batch:
            prompt_index = batch.non_tensor_batch["index"]
        else:
            prompt_index = np.array([
                ei["index"] if isinstance(ei, dict) else int(ei)
                for ei in batch.non_tensor_batch["extra_info"]
            ])
        scores = reward_tensor.sum(dim=-1)
        id2scores = defaultdict(list)
        id2prompt_key = {}
        uid = batch.non_tensor_batch["uid"]
        for i in range(len(uid)):
            id2scores[uid[i]].append(scores[i])
            id2prompt_key[uid[i]] = prompt_index[i]
        for u, uid_scores in id2scores.items():
            scores_t = torch.stack(uid_scores)
            batch_mean = torch.mean(scores_t).item() if len(scores_t) > 1 else 0.0
            batch_std = torch.std(scores_t).item() if len(scores_t) > 1 else 0.0
            self.prompt_memory_bank.update(id2prompt_key[u], batch_mean, batch_std)
    n_init = len(self.prompt_memory_bank)
    mean_mu0 = (
        sum(v[0] for v in self.prompt_memory_bank._bank.values()) / n_init
        if n_init > 0 else 0.0
    )
    print(f"[EMA-GRPO] Memory bank initialized: {n_init} prompts, mean μ₀={mean_mu0:.3f}")
```

**跳过条件（自动检测）：**
- `global_steps > 0`：从 checkpoint 恢复，跳过
- `len(memory_bank) > 0`：memory bank 非空，跳过

#### 3.9 `_save_checkpoint()` / `_load_checkpoint()` — 持久化 memory bank

```python
# save
if self.prompt_memory_bank is not None:
    memory_bank_path = os.path.join(local_global_step_folder, "prompt_memory_bank.json")
    self.prompt_memory_bank.save(memory_bank_path)

# load
if self.prompt_memory_bank is not None:
    memory_bank_path = os.path.join(global_step_folder, "prompt_memory_bank.json")
    if os.path.exists(memory_bank_path):
        self.prompt_memory_bank.load(memory_bank_path)
```

---

### 4. `verl/trainer/config/algorithm.py`

`AlgoConfig` 数据类新增字段：

```python
# EMA-GRPO 核心参数
# ema_alpha: 已废弃（动态 α 由 rollout_n 自动推导），保留字段仅供向后兼容
ema_alpha: float = 0.9
pre_init_memory_bank: bool = True  # 训练前 no-grad rollout 初始化所有 prompt 的 μ₀
                                   # pre_init=True 后 warmup_steps 在正常流程中不触发
ema_warmup_steps: int = 1          # 保险丝：pre_init=False 时跳过 count==0（未初始化）的情况

# 掌握度过滤
# mastery_threshold: 已废弃（由 rollout_n 推导为 (n-1)/n），保留字段仅供向后兼容
mastery_threshold: float = 1.0

# 可选扩展（默认禁用）
kl_gamma: float = 0.0            # KL 倍率系数；需配合 use_kl_in_reward=True
mastery_soft_alpha: float = 0.0  # 软性加权指数；0 = 硬截断
degradation_gamma: float = 0.0   # 退步感知重采样强度；0 = 禁用；需 dataloader_num_workers=0
```

---

### 5. `verl/trainer/config/ppo_trainer.yaml`

`algorithm:` 节新增：

```yaml
# EMA-GRPO 核心参数
ema_alpha: 0.9
# 训练前对全体训练 prompt 做 no-grad rollout 初始化 μ₀（使 warmup 在正常流程中不触发）
pre_init_memory_bank: true
# 保险丝：仅在 pre_init_memory_bank=false 时有意义（跳过 count==0 的未初始化状态）
ema_warmup_steps: 1

# batch_mean >= 此值时触发掌握度过滤（回退批次统计而非 EMA）
mastery_threshold: 1.0
# KL 增强系数；需同时开启 use_kl_in_reward: True
kl_gamma: 0.0
# >0 时启用软性加权替代硬截断
mastery_soft_alpha: 0.0
# 退步感知重采样；需 data.dataloader_num_workers=0
degradation_gamma: 0.0
```

---

## 使用方式

### 标准用法

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ema_grpo \
    algorithm.ema_warmup_steps=1 \
    algorithm.pre_init_memory_bank=true \
    actor_rollout_ref.rollout.n=8 \
    ...
```

训练开始前会自动运行一次 no-grad rollout pass 初始化所有 prompt 的 μ₀，之后直接进入 EMA 模式训练。
动态 α 和 τ 均由 `rollout.n=8` 自动推导，无需额外配置。

对 n=8：α ∈ [0.125, 0.875]，τ = 0.875（仅全部 8 个 rollout 正确时触发零梯度保护）。

### 关闭 pre-init（快速调试）

```bash
algorithm.pre_init_memory_bank=false
```
退回到第一次遇到 prompt 时 cold-start 初始化（warm-up 后立即生效）。

---

## 前置条件

数据集 parquet 文件的 `extra_info` 字段必须包含 `index`（整数，数据集行号）：
```python
extra_info = {"index": 0, "split": "train"}
```

验证方法：
```python
import pandas as pd
df = pd.read_parquet("~/data/processed/math500/train.parquet")
print(df["extra_info"].iloc[0])  # {"index": 0, "split": "train"}
```

---

## 实现说明

### uid 与 index 的区别

- `uid`：每训练步新生成的 `uuid4`，用于将同一 prompt 的 n 个 rollout 分组（同标准 GRPO）
- `index`：稳定的数据集行号（0…N-1），跨 epoch 不变，是 EMA bank 的可靠跨步键

### 掌握度过滤的直觉

标准 GRPO 当 n=8 rollout 全对时 `batch_std=0`，advantage=0，产生零梯度。
EMA 均值滞后会破坏这一机制（EMA 更新慢，已掌握题目的 EMA 均值 < 1.0，导致 advantage 仍为正）。

**修复方式**：先执行 `update()` 得到 $\mu_t$，再用 $\mu_t$ 与 $\tau$ 比较触发过滤。

$$\text{filter} \iff \mu_t(p) > \tau = \frac{n-1}{n}$$

用 $\mu_t$ 而非 $\hat{\mu}_t$ 的原因：
- $\hat{\mu}_t$ 是单批次估计，容易因随机性误触发（n=8 全对概率 $0.5^8 \approx 0.4\%$，训练中偶发）
- $\mu_t$ 积累了多步历史，只有模型**持续**掌握该题才会超过 $\tau$，过滤更可靠

对 n=8，τ = 0.875。单步全对后 $\mu_t$ 的变化示例（冷启动后 $\mu_0 = 0.5$）：
- 第 1 次全对：$\mu_1 \approx 0.5 \cdot 0.875 + 1.0 \cdot 0.125 = 0.56$，不触发
- 多步持续全对后 $\mu_t \to 1.0$，超过 0.875 时触发

### pre_init 与 warmup_steps=1 的关系

- `pre_init_memory_bank=true`：所有 prompt 在训练开始前 count=1，warmup 保护立即解除
- `warmup_steps=1`：作为安全后备，防止 pre-init 被跳过时 count=0 的情况
- 正常流程下 pre-init 后 warmup 不再被触发

### 内存开销

400 个 prompt × 2 个浮点数 ≈ 可忽略（~6 KB）。百万 prompt 级别无压力。
