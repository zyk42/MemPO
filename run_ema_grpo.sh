#!/usr/bin/env bash
# EMA-GRPO: cross-epoch EMA advantage normalization experiment
# Based on: experiments/qwen3-1.7b-math500-grpo/run.sh
# Differs from baseline: algorithm.adv_estimator=ema_grpo, ema_alpha, ema_warmup_steps
set -x

EXPERIMENT_DIR="$HOME/verl/experiments/grpo-forgetting-research"
BASELINE_DIR="$HOME/verl/experiments/qwen3-1.7b-math500-grpo"

# --- variant control: override via env or CLI args ---
EMA_ALPHA="${EMA_ALPHA:-0.9}"
EMA_WARMUP="${EMA_WARMUP:-3}"
EXP_SUFFIX="${EXP_SUFFIX:-alpha${EMA_ALPHA}_warmup${EMA_WARMUP}}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ema_grpo \
    algorithm.ema_alpha=${EMA_ALPHA} \
    algorithm.ema_warmup_steps=${EMA_WARMUP} \
    \
    data.train_files=$HOME/data/processed/math500/train_.parquet \
    data.val_files=$HOME/data/processed/math500/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path=$HOME/model/Qwen3-1.7B \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=10 \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.project_name='verl_grpo_math500_10epochs' \
    trainer.experiment_name="qwen3_1.7b_ema_grpo_lora_math500_${EXP_SUFFIX}" \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_logs_${EXP_SUFFIX} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation_logs_${EXP_SUFFIX} \
    trainer.default_local_dir=${EXPERIMENT_DIR}/checkpoints_${EXP_SUFFIX} \
    $@
