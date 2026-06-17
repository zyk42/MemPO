#!/usr/bin/env bash
# Qwen2.5-Math-7B × MATH-500（训练集）
# Phase A: GRPO baseline
# Phase B: EMA-GRPO (coldstart, no-std, mastery filter)
# Phase C: Dr.GRPO
# Phase D: DAPO
# Phase E: Convert + Eval on AIME2024 & AIME2025
#
# Training config:
#   dataset: 500 samples, batch_size=32 → 15 steps/epoch
#   15 epochs → 225 total steps
#   save_freq=75 → checkpoints at 75, 150, 225
#   tp=2 for vLLM rollout (7B model)
#   val: aime2024 (in-training monitor), full eval: aime2024 + aime2025
set -e
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

EXPR_DIR="$HOME/verl/experiments/grpo-forgetting-research"
TOOLS_DIR="$EXPR_DIR/tools"
MNT_DIR="/mnt/lisiqi23/grpo-forgetting-research"

RESULTS_DIR="$MNT_DIR/eval_results_qwen25math7b_math500"
MERGED_DIR="$MNT_DIR/hf_models_qwen25math7b_math500"
CKPT_GRPO="$MNT_DIR/checkpoints_qwen25math7b_math500_grpo"
CKPT_EMA="$MNT_DIR/checkpoints_qwen25math7b_math500_emagrpo"
CKPT_DRGRPO="$MNT_DIR/checkpoints_qwen25math7b_math500_drgrpo"
CKPT_DAPO="$MNT_DIR/checkpoints_qwen25math7b_math500_dapo"

mkdir -p "$RESULTS_DIR" "$MERGED_DIR"

MODEL_PATH="/mnt/lisiqi23/models/Qwen2.5-Math-7B"

# ─────────────────────────────────────────────────────────────────
# Shared trainer args
# ─────────────────────────────────────────────────────────────────
common_args() {
    echo \
    data.train_files=$HOME/data/processed/math500/all.parquet \
    data.val_files=$HOME/data/processed/aime2024_full/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=75 \
    trainer.test_freq=3 \
    trainer.project_name='verl_grpo_math500_qwen25math7b'
}

# ─────────────────────────────────────────────────────────────────
# Phase A: GRPO baseline
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase A: Training Qwen2.5-Math-7B GRPO baseline"
echo "========================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    $(common_args) \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.experiment_name='qwen25math7b_math500_grpo' \
    trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen25math7b_math500_grpo \
    trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen25math7b_math500_grpo \
    trainer.default_local_dir=${CKPT_GRPO}

echo "Phase A done."

# ─────────────────────────────────────────────────────────────────
# Phase B: EMA-GRPO (coldstart, no-std, mastery filter)
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase B: Training Qwen2.5-Math-7B EMA-GRPO (coldstart)"
echo "========================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ema_grpo \
    algorithm.ema_alpha=0.9 \
    algorithm.mastery_threshold=1.0 \
    algorithm.kl_gamma=0.0 \
    $(common_args) \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.experiment_name='qwen25math7b_math500_emagrpo' \
    trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen25math7b_math500_emagrpo \
    trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen25math7b_math500_emagrpo \
    trainer.default_local_dir=${CKPT_EMA}

echo "Phase B done."

# ─────────────────────────────────────────────────────────────────
# Phase C: Dr.GRPO (no std normalization)
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase C: Training Qwen2.5-Math-7B Dr.GRPO"
echo "========================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    $(common_args) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.experiment_name='qwen25math7b_math500_drgrpo' \
    trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen25math7b_math500_drgrpo \
    trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen25math7b_math500_drgrpo \
    trainer.default_local_dir=${CKPT_DRGRPO}

echo "Phase C done."

# ─────────────────────────────────────────────────────────────────
# Phase D: DAPO
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase D: Training Qwen2.5-Math-7B DAPO"
echo "========================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric=acc \
    $(common_args) \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.experiment_name='qwen25math7b_math500_dapo' \
    trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_qwen25math7b_math500_dapo \
    trainer.validation_data_dir=${EXPR_DIR}/validation_logs_qwen25math7b_math500_dapo \
    trainer.default_local_dir=${CKPT_DAPO}

echo "Phase D done."

# ─────────────────────────────────────────────────────────────────
# Phase E: Convert + Eval on AIME2024 & AIME2025
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Phase E: Converting and evaluating all checkpoints"
echo "========================================================"

hf_dir_of() { echo "$MERGED_DIR/${1}_step${2}/global_step_${2}/hf"; }

convert_one() {
    local CKPT_DIR="$1" LABEL="$2" STEP="$3"
    local HF_DIR; HF_DIR=$(hf_dir_of "$LABEL" "$STEP")
    [ -d "$HF_DIR" ] && { echo ">>> SKIP convert ${LABEL}_step${STEP}"; return 0; }
    echo ">>> Converting ${LABEL}_step${STEP} ..."
    CUDA_VISIBLE_DEVICES="" python3 "$TOOLS_DIR/convert_checkpoint.py" \
        --checkpoint_dir "$CKPT_DIR/global_step_${STEP}/actor" \
        --output_dir "$MERGED_DIR/${LABEL}_step${STEP}" \
        --merge_lora --lora_alpha 32
}

eval_one_gpu() {
    local GPU_ID="$1" LABEL="$2" HF_DIR="$3"
    local RESULT_FILE="$RESULTS_DIR/${LABEL}.json"
    [ -f "$RESULT_FILE" ] && { echo ">>> SKIP eval $LABEL"; return 0; }
    echo ">>> GPU $GPU_ID  evaluating $LABEL ..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "$TOOLS_DIR/eval_passk_final.py" \
        --model_dir "$HF_DIR" \
        --output "$RESULT_FILE" \
        --datasets aime2024 aime2025 \
        --n 16 --k 1 4 8 16 \
        --temperature 0.6 \
        --tp 1 --gpu_mem 0.85 \
        --max_tokens 8192
    echo ">>> GPU $GPU_ID  done: $LABEL"
}

detect_steps() { ls "$1" 2>/dev/null | grep "global_step_" | sed 's/global_step_//' | sort -n; }

# Convert
for STEP in $(detect_steps "$CKPT_GRPO"); do
    convert_one "$CKPT_GRPO" "grpo" "$STEP"
done
for STEP in $(detect_steps "$CKPT_EMA"); do
    convert_one "$CKPT_EMA" "emagrpo" "$STEP"
done
for STEP in $(detect_steps "$CKPT_DRGRPO"); do
    convert_one "$CKPT_DRGRPO" "drgrpo" "$STEP"
done
for STEP in $(detect_steps "$CKPT_DAPO"); do
    convert_one "$CKPT_DAPO" "dapo" "$STEP"
done
echo "All conversions done."

# Eval: tp=2, 4 jobs at a time (GPU pairs 0-1, 2-3, 4-5, 6-7)
LABELS=(); HF_DIRS=()
for STEP in $(detect_steps "$CKPT_GRPO"); do
    LABELS+=("grpo_step${STEP}")
    HF_DIRS+=("$(hf_dir_of grpo $STEP)")
done
for STEP in $(detect_steps "$CKPT_EMA"); do
    LABELS+=("emagrpo_step${STEP}")
    HF_DIRS+=("$(hf_dir_of emagrpo $STEP)")
done
for STEP in $(detect_steps "$CKPT_DRGRPO"); do
    LABELS+=("drgrpo_step${STEP}")
    HF_DIRS+=("$(hf_dir_of drgrpo $STEP)")
done
for STEP in $(detect_steps "$CKPT_DAPO"); do
    LABELS+=("dapo_step${STEP}")
    HF_DIRS+=("$(hf_dir_of dapo $STEP)")
done

N_JOBS=${#LABELS[@]}
echo "Total eval jobs: $N_JOBS"

i=0
while [ $i -lt $N_JOBS ]; do
    PIDS=()
    for slot in 0 1 2 3; do
        idx=$((i + slot))
        [ $idx -ge $N_JOBS ] && break
        GPU_START=$slot
        eval_one_gpu "$GPU_START" "${LABELS[$idx]}" "${HF_DIRS[$idx]}" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    i=$((i + 4))  # 4 concurrent eval jobs (one per GPU, tp=1)
done

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS: Qwen2.5-Math-7B × MATH-500 train / AIME eval"
echo "============================================================"
python3 - <<'EOF'
import json, glob, os
results_dir = "/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research/eval_results_qwen25math7b_math500"
files = sorted(glob.glob(f"{results_dir}/*.json"))
if not files:
    print("No results found."); exit()
DATASETS = ["aime2024", "aime2025"]
KS = [1, 4, 8, 16]
hdr = f"{'Model':<40}"
for ds in DATASETS:
    for k in KS:
        hdr += f"  {ds}/p@{k}"
print(hdr)
print("-" * len(hdr))
for fpath in files:
    with open(fpath) as f:
        data = json.load(f)
    name = os.path.basename(fpath).replace(".json", "")
    row = f"{name:<40}"
    for ds in DATASETS:
        for k in KS:
            v = data.get("datasets", {}).get(ds, {}).get("pass_at_k", {}).get(f"pass@{k}", float("nan"))
            row += f"  {v:.3f}"
    print(row)
EOF

echo "All done. Results at: $RESULTS_DIR"
