#!/usr/bin/env bash
# Qwen3-1.7B-Base × MATH-500 训练
# Phase A: GRPO baseline
# Phase B: EMA-GRPO (dynamic alpha, pre_init_memory_bank=false)
#
# 训练配置：
#   dataset: 500 samples, batch_size=32 → ~15 steps/epoch
#   15 epochs → 225 total steps
#   save_freq=75 → checkpoints at 75, 150, 225
#   val: aime2024 (in-training monitor)
#   tp=1, 4 GPUs
set -e
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

EXPR_DIR="$HOME/verl/experiments/grpo-forgetting-research"
TOOLS_DIR="$EXPR_DIR/tools"
MNT_DIR="/mnt/lisiqi23/grpo-forgetting-research"

CKPT_GRPO="$MNT_DIR/checkpoints_qwen3_1.7b_base_grpo"
CKPT_EMA="$MNT_DIR/checkpoints_qwen3_1.7b_base_emagrpo"

MODEL_PATH="/mnt/lisiqi23/models/Qwen3-1.7B-Base"
TRAIN_DATA="$HOME/data/processed/math500/all.parquet"
MEMORY_BANK_PATH="$MNT_DIR/math500_memory_bank_qwen3_1.7b_base.json"

mkdir -p "$CKPT_GRPO" "$CKPT_EMA"

# ─────────────────────────────────────────────────────────────────
# Phase A: GRPO
# ─────────────────────────────────────────────────────────────────
if [ -d "$CKPT_GRPO/global_step_225" ]; then
    echo ">>> SKIP Phase A: GRPO checkpoint exists"
else
    echo "========================================================"
    echo "Phase A: Training GRPO on Qwen3-1.7B-Base"
    echo "========================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        \
        data.train_files=${TRAIN_DATA} \
        data.val_files=$HOME/data/processed/aime2024_full/test.parquet \
        data.train_batch_size=32 \
        data.max_prompt_length=512 \
        data.max_response_length=12288 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.model.lora_rank=64 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_num_seqs=256 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        \
        algorithm.use_kl_in_reward=False \
        \
        trainer.val_before_train=True \
        trainer.critic_warmup=0 \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.total_epochs=15 \
        trainer.save_freq=75 \
        trainer.test_freq=75 \
        trainer.resume_mode=auto \
        trainer.project_name='verl_math500_qwen3_1.7b_base' \
        trainer.experiment_name='qwen3_1.7b_base_grpo' \
        trainer.logger='["console","swanlab"]' \
        trainer.log_val_generations=10 \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen3_1.7b_base_grpo \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen3_1.7b_base_grpo \
        trainer.default_local_dir=${CKPT_GRPO}

    echo "Phase A done."
fi

# ─────────────────────────────────────────────────────────────────
# Phase B: EMA-GRPO (dynamic alpha)
# ─────────────────────────────────────────────────────────────────
if [ -d "$CKPT_EMA/global_step_225" ]; then
    echo ">>> SKIP Phase B: EMA-GRPO checkpoint exists"
else
    # Step 1: 预计算 PromptMemoryBank μ₀
    if [ -f "$MEMORY_BANK_PATH" ]; then
        echo "Memory bank exists at $MEMORY_BANK_PATH, skipping precompute."
    else
        echo "Pre-computing PromptMemoryBank μ₀ via standalone vLLM ..."
        python3 "$TOOLS_DIR/precompute_memory_bank.py" \
            --train_file "$TRAIN_DATA" \
            --model_dir "$MODEL_PATH" \
            --output "$MEMORY_BANK_PATH" \
            --n 8 \
            --temperature 1.0 \
            --max_tokens 12288 \
            --tp 1 \
            --gpu_mem 0.90
        echo "Memory bank saved to $MEMORY_BANK_PATH"
    fi

    echo "========================================================"
    echo "Phase B: Training EMA-GRPO (dynamic alpha) on Qwen3-1.7B-Base"
    echo "========================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=ema_grpo \
        algorithm.pre_init_memory_bank=false \
        algorithm.memory_bank_init_path=${MEMORY_BANK_PATH} \
        algorithm.kl_gamma=0.0 \
        algorithm.use_kl_in_reward=False \
        \
        data.train_files=${TRAIN_DATA} \
        data.val_files=$HOME/data/processed/aime2024_full/test.parquet \
        data.train_batch_size=32 \
        data.max_prompt_length=512 \
        data.max_response_length=12288 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.model.lora_rank=64 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_num_seqs=256 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        \
        trainer.val_before_train=True \
        trainer.critic_warmup=0 \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.total_epochs=15 \
        trainer.save_freq=75 \
        trainer.test_freq=75 \
        trainer.resume_mode=auto \
        trainer.project_name='verl_math500_qwen3_1.7b_base' \
        trainer.experiment_name='qwen3_1.7b_base_emagrpo' \
        trainer.logger='["console","swanlab"]' \
        trainer.log_val_generations=10 \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen3_1.7b_base_emagrpo \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen3_1.7b_base_emagrpo \
        trainer.default_local_dir=${CKPT_EMA}

    echo "Phase B done."
fi

echo "All training done."
