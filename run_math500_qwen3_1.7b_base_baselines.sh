#!/usr/bin/env bash
# Qwen3-1.7B-Base × MATH-500 训练：DAPO 和 DRGRPO
# 使用方法：
#   bash run_math500_qwen3_1.7b_base_baselines.sh drgrpo  # 只跑 DRGRPO
#   bash run_math500_qwen3_1.7b_base_baselines.sh dapo   # 只跑 DAPO
#   bash run_math500_qwen3_1.7b_base_baselines.sh all    # 全部跑
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

MODEL_PATH="/mnt/lisiqi23/models/Qwen3-1.7B-Base"
TRAIN_DATA="$HOME/data/processed/math500/all.parquet"

CKPT_DRGRPO="$MNT_DIR/checkpoints_qwen3_1.7b_base_drgrpo"
CKPT_DAPO="$MNT_DIR/checkpoints_qwen3_1.7b_base_dapo"
CKPT_REINFORCEPP="$MNT_DIR/checkpoints_qwen3_1.7b_base_reinforcepp"

mkdir -p "$CKPT_DRGRPO" "$CKPT_DAPO" "$CKPT_REINFORCEPP"

# ─────────────────────────────────────────────────────────────────
# DRGRPO
# 唯一区别：norm_adv_by_std_in_grpo=False（不除以 std）
# 论文：https://arxiv.org/abs/2503.20783
# ─────────────────────────────────────────────────────────────────
run_drgrpo() {
    if [ -d "$CKPT_DRGRPO/global_step_225" ]; then
        echo ">>> SKIP DRGRPO: checkpoint exists"
        return
    fi
    echo "========================================================"
    echo "DRGRPO on Qwen3-1.7B-Base"
    echo "========================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=False \
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
        trainer.experiment_name='qwen3_1.7b_base_drgrpo' \
        trainer.logger='["console","swanlab"]' \
        trainer.log_val_generations=10 \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen3_1.7b_base_drgrpo \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen3_1.7b_base_drgrpo \
        trainer.default_local_dir=${CKPT_DRGRPO}

    echo "DRGRPO done."
}

# ─────────────────────────────────────────────────────────────────
# DAPO
# 与 GRPO 的五点区别（来自 DAPO 论文）：
#   1. 过滤全对/全错 group（filter_groups.enable=True）
#   2. 不除 std（norm_adv_by_std_in_grpo=False）
#   3. Clip-higher：正 advantage 用更大 clip（0.28 vs 0.2）
#   4. Token-level entropy bonus（entropy_coeff=0.001）
#   5. 去掉 KL loss（DAPO 完全不用 KL 约束）
# ─────────────────────────────────────────────────────────────────
run_dapo() {
    if [ -d "$CKPT_DAPO/global_step_225" ]; then
        echo ">>> SKIP DAPO: checkpoint exists"
        return
    fi
    echo "========================================================"
    echo "DAPO on Qwen3-1.7B-Base"
    echo "========================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=False \
        +algorithm.filter_groups.enable=True \
        +algorithm.filter_groups.metric=acc \
        +algorithm.filter_groups.max_num_gen_batches=0 \
        \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
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
        trainer.experiment_name='qwen3_1.7b_base_dapo' \
        trainer.logger='["console","swanlab"]' \
        trainer.log_val_generations=10 \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen3_1.7b_base_dapo \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen3_1.7b_base_dapo \
        trainer.default_local_dir=${CKPT_DAPO}

    echo "DAPO done."
}

# ─────────────────────────────────────────────────────────────────
# REINFORCE++
# ─────────────────────────────────────────────────────────────────
run_reinforcepp() {
    if [ -d "$CKPT_REINFORCEPP/global_step_225" ]; then
        echo ">>> SKIP REINFORCE++: checkpoint exists"
        return
    fi
    echo "========================================================"
    echo "REINFORCE++ on Qwen3-1.7B-Base"
    echo "========================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=reinforce_plus_plus \
        \
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
        trainer.experiment_name='qwen3_1.7b_base_reinforcepp' \
        trainer.logger='["console","swanlab"]' \
        trainer.log_val_generations=10 \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen3_1.7b_base_reinforcepp \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen3_1.7b_base_reinforcepp \
        trainer.default_local_dir=${CKPT_REINFORCEPP}

    echo "REINFORCE++ done."
}

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
TARGET="${1:-all}"

case "$TARGET" in
    drgrpo)   run_drgrpo ;;
    dapo)     run_dapo ;;
    reinforcepp) run_reinforcepp ;;
    all)
        run_drgrpo
        echo ""
        echo "DRGRPO done. Starting DAPO..."
        echo ""
        run_dapo
        echo ""
        echo "DAPO done. Starting REINFORCE++..."
        echo ""
        run_reinforcepp
        ;;
    *)
        echo "Usage: $0 [drgrpo|dapo|reinforcepp|all]"
        exit 1 ;;
esac

echo "All training done."