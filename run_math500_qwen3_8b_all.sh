#!/usr/bin/env bash
# Qwen3-8B × MATH-500：GRPO / EMA-GRPO / DAPO / DrGRPO 顺序训练
#
# 训练参数：
#   dataset   : math500 (500 samples, all.parquet)
#   batch     : 32 → ~15 steps/epoch → 15 epochs ≈ 225 total steps
#   save      : every 75 steps (5 epochs) → checkpoints at 75 / 150 / 225
#   val       : aime2024 + aime2025，every 3 steps
#   max_tokens: 12288 (thinking model)
#   hardware  : 4 × H20 (96GB)，vLLM tp=1（单卡可容纳 8B）
#
# 用法：
#   bash run_math500_qwen3_8b_all.sh          # 顺序跑全部 4 个方法 + 转换评测
#   bash run_math500_qwen3_8b_all.sh grpo     # 只跑 GRPO
#   bash run_math500_qwen3_8b_all.sh emagrpo  # 只跑 EMA-GRPO
#   bash run_math500_qwen3_8b_all.sh dapo     # 只跑 DAPO
#   bash run_math500_qwen3_8b_all.sh drgrpo   # 只跑 DrGRPO
#   bash run_math500_qwen3_8b_all.sh eval     # 只做 convert + eval（跳过训练）

set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXPR_DIR="$HOME/verl/experiments/grpo-forgetting-research"
TOOLS_DIR="$EXPR_DIR/tools"
MNT_DIR="/mnt/lisiqi23/grpo-forgetting-research"

MODEL_PATH="/mnt/lisiqi23/models/Qwen3-8B"

RESULTS_DIR="$MNT_DIR/eval_results_math500_qwen3_8b"
MERGED_DIR="$MNT_DIR/hf_models_math500_qwen3_8b"

mkdir -p "$RESULTS_DIR" "$MERGED_DIR"

TARGET="${1:-all}"

# ──────────────────────────────────────────────────────────────────────────────
# 公共训练参数
# ──────────────────────────────────────────────────────────────────────────────
COMMON_ARGS="
    data.train_files=$HOME/data/processed/math500/all.parquet
    data.val_files=[$HOME/data/processed/aime2024_full/test.parquet,$HOME/data/processed/aime2025_full/test.parquet]
    data.train_batch_size=32
    data.max_prompt_length=512
    data.max_response_length=10240
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True

    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.lora_rank=64
    actor_rollout_ref.model.lora_alpha=32
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    algorithm.use_kl_in_reward=False

    trainer.val_before_train=True
    trainer.critic_warmup=0
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.total_epochs=15
    trainer.save_freq=75
    trainer.test_freq=75
    trainer.logger=[\"console\",\"swanlab\"]
    trainer.log_val_generations=10
    trainer.resume_mode=auto
"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: GRPO
# adv = (r - mean) / std，KL loss on
# ──────────────────────────────────────────────────────────────────────────────
run_grpo() {
    EXP="qwen3_8b_math500_grpo"
    CKPT="$MNT_DIR/checkpoints_${EXP}"
    echo "========================================================"
    echo ">>> [1/4] GRPO"
    echo "========================================================"
    set -x
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        ${COMMON_ARGS} \
        trainer.project_name='verl_math500_qwen3_8b' \
        trainer.experiment_name="${EXP}" \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_${EXP} \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_${EXP} \
        trainer.default_local_dir=${CKPT}
    { set +x; } 2>/dev/null
    echo ">>> GRPO done."
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: EMA-GRPO（coldstart，无 std，mastery filter）
# adv = r - μ_EMA，α=0.9，mastery_threshold=1.0
# ──────────────────────────────────────────────────────────────────────────────
run_emagrpo() {
    EXP="qwen3_8b_math500_emagrpo"
    CKPT="$MNT_DIR/checkpoints_${EXP}"
    echo "========================================================"
    echo ">>> [2/4] EMA-GRPO"
    echo "========================================================"
    set -x
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=ema_grpo \
        algorithm.ema_alpha=0.9 \
        algorithm.mastery_threshold=1.0 \
        algorithm.kl_gamma=0.0 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        ${COMMON_ARGS} \
        trainer.project_name='verl_math500_qwen3_8b' \
        trainer.experiment_name="${EXP}" \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_${EXP} \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_${EXP} \
        trainer.default_local_dir=${CKPT}
    { set +x; } 2>/dev/null
    echo ">>> EMA-GRPO done."
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: DAPO
# 全对/全错 group 过滤 + asymmetric clip（0.2/0.28）+ entropy bonus，无 KL loss
# ──────────────────────────────────────────────────────────────────────────────
run_dapo() {
    EXP="qwen3_8b_math500_dapo"
    CKPT="$MNT_DIR/checkpoints_${EXP}"
    echo "========================================================"
    echo ">>> [3/4] DAPO"
    echo "========================================================"
    set -x
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=False \
        +algorithm.filter_groups.enable=True \
        +algorithm.filter_groups.metric=acc \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        ${COMMON_ARGS} \
        trainer.project_name='verl_math500_qwen3_8b' \
        trainer.experiment_name="${EXP}" \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_${EXP} \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_${EXP} \
        trainer.default_local_dir=${CKPT}
    { set +x; } 2>/dev/null
    echo ">>> DAPO done."
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: DrGRPO（去掉 std 归一化，其余与 GRPO 一致）
# adv = r - mean（无 std），KL loss on
# ──────────────────────────────────────────────────────────────────────────────
run_drgrpo() {
    EXP="qwen3_8b_math500_drgrpo"
    CKPT="$MNT_DIR/checkpoints_${EXP}"
    echo "========================================================"
    echo ">>> [4/4] DrGRPO"
    echo "========================================================"
    set -x
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=False \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        ${COMMON_ARGS} \
        trainer.project_name='verl_math500_qwen3_8b' \
        trainer.experiment_name="${EXP}" \
        trainer.rollout_data_dir=${EXPR_DIR}/rollout_logs_${EXP} \
        trainer.validation_data_dir=${EXPR_DIR}/validation_logs_${EXP} \
        trainer.default_local_dir=${CKPT}
    { set +x; } 2>/dev/null
    echo ">>> DrGRPO done."
}

# ──────────────────────────────────────────────────────────────────────────────
# Convert + Eval
# tp=1，4 GPU 各跑一个 job（单卡评 8B，max_tokens=12288）
# 评测集：math500 + aime2024 + aime2025 + amc，n=8
# ──────────────────────────────────────────────────────────────────────────────
run_eval() {
    echo "========================================================"
    echo ">>> Convert + Eval"
    echo "========================================================"

    METHODS=(grpo emagrpo dapo drgrpo)

    # ── Convert ──────────────────────────────────────────────────
    for METHOD in "${METHODS[@]}"; do
        EXP="qwen3_8b_math500_${METHOD}"
        CKPT="$MNT_DIR/checkpoints_${EXP}"
        for STEP_DIR in "$CKPT"/global_step_*/; do
            [ -d "$STEP_DIR" ] || continue
            STEP=$(basename "$STEP_DIR" | sed 's/global_step_//')
            HF_DIR="$MERGED_DIR/${METHOD}_step${STEP}/global_step_${STEP}/hf"
            if [ -d "$HF_DIR" ]; then
                echo "SKIP convert ${METHOD}_step${STEP}"
                continue
            fi
            echo "Converting ${METHOD}_step${STEP} ..."
            CUDA_VISIBLE_DEVICES="" python3 "$TOOLS_DIR/convert_checkpoint.py" \
                --checkpoint_dir "$STEP_DIR/actor" \
                --output_dir "$MERGED_DIR/${METHOD}_step${STEP}" \
                --merge_lora --lora_alpha 32
        done
    done
    echo "All conversions done."

    # ── Build eval job list ───────────────────────────────────────
    LABELS=()
    HF_DIRS=()
    for METHOD in "${METHODS[@]}"; do
        EXP="qwen3_8b_math500_${METHOD}"
        CKPT="$MNT_DIR/checkpoints_${EXP}"
        for STEP_DIR in "$CKPT"/global_step_*/; do
            [ -d "$STEP_DIR" ] || continue
            STEP=$(basename "$STEP_DIR" | sed 's/global_step_//')
            LABELS+=("${METHOD}_step${STEP}")
            HF_DIRS+=("$MERGED_DIR/${METHOD}_step${STEP}/global_step_${STEP}/hf")
        done
    done

    # base model
    LABELS+=("base")
    HF_DIRS+=("$MODEL_PATH")

    N_JOBS=${#LABELS[@]}
    echo "Total eval jobs: $N_JOBS"

    eval_one() {
        local GPU="$1" LABEL="$2" HF_DIR="$3"
        local RESULT="$RESULTS_DIR/${LABEL}.json"
        if [ -f "$RESULT" ]; then echo "SKIP eval $LABEL"; return 0; fi
        echo "[GPU $GPU] eval $LABEL ..."
        CUDA_VISIBLE_DEVICES="$GPU" python3 "$TOOLS_DIR/eval_passk_final.py" \
            --model_dir "$HF_DIR" \
            --output "$RESULT" \
            --datasets math500 aime2024 aime2025 amc \
            --n 8 --k 1 4 8 \
            --temperature 0.6 \
            --tp 1 --gpu_mem 0.85 \
            --max_tokens 10240
        echo "[GPU $GPU] done: $LABEL"
    }

    # 4 jobs 并行（GPU 0-3，每个 job 独占 1 张）
    i=0
    while [ $i -lt $N_JOBS ]; do
        PIDS=()
        for slot in 0 1 2 3; do
            idx=$((i + slot))
            [ $idx -ge $N_JOBS ] && break
            eval_one "$slot" "${LABELS[$idx]}" "${HF_DIRS[$idx]}" &
            PIDS+=($!)
        done
        wait "${PIDS[@]}"
        i=$((i + 4))
    done

    # ── Summary table ─────────────────────────────────────────────
    echo ""
    echo "================================================================"
    echo "  RESULTS: Qwen3-8B × MATH-500"
    echo "================================================================"
    python3 - <<'PYEOF'
import json, glob, os

results_dir = "/mnt/lisiqi23/grpo-forgetting-research/eval_results_math500_qwen3_8b"
files = sorted(glob.glob(f"{results_dir}/*.json"))
if not files:
    print("No results found."); exit()

DATASETS = ["math500", "aime2024", "aime2025", "amc"]
KS = [1, 4, 8]

hdr = f"{'Model':<35}"
for ds in DATASETS:
    for k in KS:
        hdr += f"  {ds}/p@{k}"
print(hdr)
print("-" * len(hdr))

ORDER = [
    "base",
    "grpo_step75",    "grpo_step150",    "grpo_step225",
    "emagrpo_step75", "emagrpo_step150", "emagrpo_step225",
    "dapo_step75",    "dapo_step150",    "dapo_step225",
    "drgrpo_step75",  "drgrpo_step150",  "drgrpo_step225",
]
file_map = {os.path.basename(f).replace(".json", ""): f for f in files}

for name in ORDER:
    fpath = file_map.get(name)
    if not fpath:
        continue
    with open(fpath) as f:
        data = json.load(f)
    row = f"{name:<35}"
    for ds in DATASETS:
        for k in KS:
            v = data.get("datasets", {}).get(ds, {}).get("pass_at_k", {}).get(f"pass@{k}", float("nan"))
            row += f"  {v:.3f}"
    print(row)

for name, fpath in sorted(file_map.items()):
    if name not in ORDER:
        with open(fpath) as f:
            data = json.load(f)
        row = f"{name:<35}"
        for ds in DATASETS:
            for k in KS:
                v = data.get("datasets", {}).get(ds, {}).get("pass_at_k", {}).get(f"pass@{k}", float("nan"))
                row += f"  {v:.3f}"
        print(row)
PYEOF
}

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────
case "$TARGET" in
    grpo)    run_grpo ;;
    emagrpo) run_emagrpo ;;
    dapo)    run_dapo ;;
    drgrpo)  run_drgrpo ;;
    eval)    run_eval ;;
    all)
        run_grpo
        run_emagrpo
        run_dapo
        run_drgrpo
        run_eval
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [grpo|emagrpo|dapo|drgrpo|eval|all]"
        exit 1
        ;;
esac

echo ""
echo "All done. Results: $RESULTS_DIR"
