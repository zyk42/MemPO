# 交接文档：GRPO 防遗忘研究

> 最后更新：2026-05-15
> 工作目录：`/root/verl/experiments/grpo-forgetting-research/`
> 挂载卷：`/mnt/lisiqi23/grpo-forgetting-research/`（checkpoint/eval 均写此处）

---

## 1. 当前进行中的任务

### 🔄 正在训练：Qwen3-8B × MATH-500 — Phase A (GRPO)

```
PID:    1844839
启动:   2026-05-15 19:16
日志:   /root/verl/experiments/grpo-forgetting-research/rollout_logs_qwen3_8b_math500_grpo/
模型:   /mnt/lisiqi23/models/Qwen3-8B
ckpt:   /mnt/lisiqi23/grpo-forgetting-research/checkpoints_qwen3_8b_math500_grpo/
SwanLab: project=verl_math500_qwen3_8b, experiment=qwen3_8b_math500_grpo
```

**关键配置（与 run_math500_qwen3_8b.sh 脚本有差异）**：

| 参数 | 实际值 | 脚本值 |
|------|--------|--------|
| `tensor_model_parallel_size` | 1 | 2 |
| `n_gpus_per_node` | 4 | 8 |
| `max_response_length` | 10240 | 8192 |
| `test_freq` | 75 | 3 |
| `resume_mode` | auto | — |
| val_files | aime2024 + aime2025 | aime2024 |

**完成后需手动启动 Phase B (EMA-GRPO)**（见第 3 节命令）

---

## 2. Qwen2.5-Math-7B × MATH-500 实验状态（已全部完成）

训练脚本：`run_math500_qwen25math7b.sh`

| Phase | 方法 | 状态 | Checkpoint（step）|
|-------|------|------|-------------------|
| A | GRPO | ✅ 完成 | 75 / 150 / 225 |
| B | EMA-GRPO | ✅ 完成 | 75 / 150 / 225 |
| C | Dr.GRPO | ✅ 完成（崩在 05-15 00:19，ckpt 完整）| 75 / 150 / 225 |
| D | DAPO | ✅ 完成（崩在 05-15 08:18，ckpt 完整）| 75 / 150 / 225 |
| E | Convert + Eval | ⚠️ 部分完成 | step150/225 已有，step75 缺失 |

所有路径前缀：`/mnt/lisiqi23/grpo-forgetting-research/`

### Eval 结果（AIME2024 pass@1 / pass@8）

| 方法 | step150 p@1 | step150 p@8 | step225 p@1 | step225 p@8 |
|------|-------------|-------------|-------------|-------------|
| Base（无训练）| 0.163 | 0.384 | — | — |
| GRPO | 0.210 | 0.475 | 0.213 | 0.445 |
| EMA-GRPO | 0.219 | 0.431 | **0.227** | **0.526** |
| Dr.GRPO | 0.194 | 0.432 | 0.213 | 0.452 |
| DAPO | 0.227 | 0.454 | **0.258** | **0.506** |

> **结论**：step225 上 DAPO 最高（AIME2024 p@1=0.258），EMA-GRPO p@8 最高（0.526）。

<details>
<summary>AIME2025 / AMC / MATH-500 完整数据</summary>

| 方法 | step | AIME2025 p@1 | AIME2025 p@8 | AMC p@1 | MATH500 p@1 |
|------|------|--------------|--------------|---------|-------------|
| Base | — | 0.075 | 0.192 | 0.447 | 0.594 |
| GRPO | 150 | 0.108 | 0.206 | 0.578 | 0.716 |
| GRPO | 225 | 0.090 | 0.207 | 0.572 | 0.755 |
| EMA-GRPO | 150 | 0.092 | 0.229 | 0.588 | 0.727 |
| EMA-GRPO | 225 | 0.106 | 0.189 | 0.588 | **0.763** |
| Dr.GRPO | 150 | 0.090 | 0.267 | 0.569 | 0.718 |
| Dr.GRPO | 225 | 0.081 | 0.206 | 0.578 | 0.744 |
| DAPO | 150 | 0.083 | 0.250 | 0.591 | 0.714 |
| DAPO | 225 | **0.121** | **0.344** | **0.597** | 0.757 |

</details>

### 未完成的 Eval（step75 缺失）

需要补跑 step75 的 convert + eval（4 个方法 × 1 步 = 4 个任务）：
```bash
# 示例：补跑 grpo step75
TOOLS_DIR=/root/verl/experiments/grpo-forgetting-research/tools
MNT=/mnt/lisiqi23/grpo-forgetting-research

CUDA_VISIBLE_DEVICES="" python3 $TOOLS_DIR/convert_checkpoint.py \
    --checkpoint_dir $MNT/checkpoints_qwen25math7b_math500_grpo/global_step_75/actor \
    --output_dir $MNT/hf_models_qwen25math7b_math500/grpo_step75 \
    --merge_lora --lora_alpha 32
# 其余方法同理，再跑 eval_passk_final.py
```

---

## 3. 待执行操作

### Step 1：等 Qwen3-8B GRPO 完成后启动 EMA-GRPO

确认完成：
```bash
ls /mnt/lisiqi23/grpo-forgetting-research/checkpoints_qwen3_8b_math500_grpo/
# 应看到 global_step_75, global_step_150, global_step_225
```

启动 EMA-GRPO（与当前 GRPO 配置对齐，使用实际参数而非脚本默认）：
```bash
cd /root/verl
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ema_grpo \
    data.train_files=$HOME/data/processed/math500/all.parquet \
    data.val_files=[$HOME/data/processed/aime2024_full/test.parquet,$HOME/data/processed/aime2025_full/test.parquet] \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=/mnt/lisiqi23/models/Qwen3-8B \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=75 \
    trainer.test_freq=75 \
    trainer.resume_mode=auto \
    trainer.project_name='verl_math500_qwen3_8b' \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.experiment_name='qwen3_8b_math500_emagrpo' \
    trainer.rollout_data_dir=/root/verl/experiments/grpo-forgetting-research/rollout_logs_qwen3_8b_math500_emagrpo \
    trainer.validation_data_dir=/root/verl/experiments/grpo-forgetting-research/validation_logs_qwen3_8b_math500_emagrpo \
    trainer.default_local_dir=/mnt/lisiqi23/grpo-forgetting-research/checkpoints_qwen3_8b_math500_emagrpo \
    > /tmp/math500_qwen3_8b_emagrpo.log 2>&1 &
```

### Step 2：两个方法完成后运行 Convert + Eval

```bash
TOOLS_DIR=/root/verl/experiments/grpo-forgetting-research/tools
MNT=/mnt/lisiqi23/grpo-forgetting-research
RESULTS_DIR=$MNT/eval_results_qwen3_8b_math500
MERGED_DIR=$MNT/hf_models_qwen3_8b_math500
mkdir -p "$RESULTS_DIR" "$MERGED_DIR"

for method in grpo emagrpo; do
    CKPT_DIR=$MNT/checkpoints_qwen3_8b_math500_${method}
    for STEP in 75 150 225; do
        CUDA_VISIBLE_DEVICES="" python3 $TOOLS_DIR/convert_checkpoint.py \
            --checkpoint_dir $CKPT_DIR/global_step_${STEP}/actor \
            --output_dir $MERGED_DIR/${method}_step${STEP} \
            --merge_lora --lora_alpha 32
    done
done

# Eval (tp=1, 4 GPUs)
for method in grpo emagrpo; do
    for STEP in 75 150 225; do
        MODEL_DIR=$MERGED_DIR/${method}_step${STEP}/global_step_${STEP}/hf
        CUDA_VISIBLE_DEVICES=0,1 python3 $TOOLS_DIR/eval_passk_final.py \
            --model_dir "$MODEL_DIR" \
            --output_file "$RESULTS_DIR/${method}_step${STEP}.json" \
            --datasets aime2024 aime2025 amc math500 \
            --n 16 --temperature 0.6 --max_tokens 10240 \
            --tensor_parallel_size 1 &
        # 配合实际可用GPU并发
    done
done
```

---

## 4. 关键配置说明

### Qwen3-8B 特殊注意事项

| 项目 | 说明 |
|------|------|
| GPU 配置 | 4 GPUs，tp=1（不是脚本里的 8 GPUs tp=2）|
| `max_response_length` | 10240（不是脚本里的 8192）|
| `test_freq` | 75（每 epoch 末验证，不是 3）|
| `kl_loss_type` | `low_var_kl`，coef=0.001 |
| `entropy_coeff` | 0（关闭熵正则）|
| `resume_mode` | auto（断点续训）|
| 模型路径 | `/mnt/lisiqi23/models/Qwen3-8B` |

### Qwen2.5-Math-7B 特殊注意事项

| 项目 | 说明 |
|------|------|
| `max_position_embeddings` | **4096**（不是 8192） |
| vLLM 启动 | **必须加** `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| LoRA alpha | 32（convert 时必须加 `--lora_alpha 32`，否则权重被清零）|
| eval `--max_tokens` | 8192 |
| vLLM tp | 2（7B 需要 tensor_parallel_size=2）|

### SwanLab 日志

- API Key：`DSIMPCMh3iRqsCL1GkE46`
- 7B 项目：`verl_grpo_math500_qwen25math7b`，实验名 `qwen25math7b_math500_{grpo,emagrpo,drgrpo,dapo}`
- 8B 项目：`verl_math500_qwen3_8b`，实验名 `qwen3_8b_math500_{grpo,emagrpo}`

---

## 5. 完整实验矩阵（全局）

| 模型 | 训练集 | 方法 | 状态 | 结果路径 |
|------|--------|------|------|---------|
| Qwen2.5-Math-1.5B | MATH-Train-2800 | GRPO/DrGRPO/DAPO/EMA-GRPO/EMA-GRPO-coldstart | ✅ 完成 | `eval_results_qwen25math_1.5b/` |
| Qwen3-1.7B | MATH-Train-2800 | GRPO / EMA-GRPO-coldstart | ✅ 完成 | `eval_results_mathtrain_qwen3_redo/` |
| Qwen2.5-Math-7B | MATH-500 | GRPO ✅ / EMA-GRPO ✅ / DrGRPO ✅ / DAPO ✅ | **eval step75 待补** | `eval_results_qwen25math7b_math500/` |
| **Qwen3-8B** | **MATH-500** | **GRPO 🔄 / EMA-GRPO ⏳** | **进行中** | `eval_results_qwen3_8b_math500/`（未创建）|

---

## 6. 关键文件路径索引

```
研究笔记（全部分析）:
  /root/verl/experiments/grpo-forgetting-research/research_notes.md

训练脚本:
  /root/verl/experiments/grpo-forgetting-research/run_math500_qwen25math7b.sh   ← 7B 已完成
  /root/verl/experiments/grpo-forgetting-research/run_math500_qwen3_8b.sh       ← 8B 参考脚本（实际参数见第3节）

工具脚本:
  /root/verl/experiments/grpo-forgetting-research/tools/convert_checkpoint.py   ← FSDP→HF 转换
  /root/verl/experiments/grpo-forgetting-research/tools/eval_passk_final.py     ← pass@k 评测

训练日志（实时）:
  rollout_logs_qwen3_8b_math500_grpo/    ← 当前进行中（8B GRPO）
  /tmp/math500_qwen25math7b_drgrpo.log   ← 7B DrGRPO（已完成，崩溃前ckpt完整）
  /tmp/math500_qwen25math7b_dapo.log     ← 7B DAPO（已完成，崩溃前ckpt完整）

数据集（已处理）:
  /root/data/processed/math500/all.parquet         ← 500 题（train+test 合并）
  /root/data/processed/aime2024_full/test.parquet  ← AIME2024（30题）
  /root/data/processed/aime2025_full/test.parquet  ← AIME2025
  /root/data/processed/amc/test.parquet            ← AMC 40 题
  /root/data/processed/math_train_2800/train.parquet

挂载卷根目录（新路径）:
  /mnt/lisiqi23/grpo-forgetting-research/
  ├── checkpoints_qwen25math7b_math500_{grpo,emagrpo,drgrpo,dapo}/  ← 全部完成
  ├── checkpoints_qwen3_8b_math500_grpo/                            ← 训练中
  ├── hf_models_qwen25math7b_math500/                               ← step150/225 已转换
  ├── eval_results_qwen25math7b_math500/                            ← step150/225 已有
  └── （eval_results_qwen3_8b_math500/ 待创建）

备份同步路径:
  /mnt/lisiqi23/ema/verl/experiments/grpo-forgetting-research/

模型路径:
  /mnt/lisiqi23/models/Qwen3-8B          ← 当前使用
  /mnt/lisiqi23/models/Qwen2.5-Math-7B
  /mnt/lisiqi23/models/Qwen3-1.7B
  /mnt/lisiqi23/models/Qwen2.5-Math-1.5B

核心实现:
  /root/verl/verl/utils/prompt_memory_bank.py      ← PromptMemoryBank
  /root/verl/verl/trainer/ppo/ray_trainer.py       ← EMA advantage 集成点
```

---

## 7. 常见问题 & 已知 Bug

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| vLLM 报 `max_model_len > max_position_embeddings` | Qwen2.5-Math-7B max_position_embeddings=4096 | 启动时加 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| LoRA 合并后权重全零 | `legacy_model_merger.py` 硬编码 `lora_alpha=0` | convert 时加 `--lora_alpha 32` |
| Qwen2.5-Math-1.5B eval CUDA assert | max_position_embeddings=4096 | eval 脚本用 `--max_tokens 4096` |
| EMA-GRPO pg_loss 爆炸 | cold-start 时 batch_std=0 导致 ema_std→1e-6 | 去掉 std 归一化，只做均值 EMA |
| Qwen3-1.7B EMA-GRPO step~1390 divergence | mastery filter 在高 epoch 导致 thinking loop | 限制 epoch ≤ 12 或增大 KL coef |
| 7B DrGRPO/DAPO 训练进程崩溃 | DataLoader worker OOM killed | Checkpoint 完整，eval 不受影响 |
