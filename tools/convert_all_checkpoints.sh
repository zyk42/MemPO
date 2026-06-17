#!/usr/bin/env bash
# 批量将所有实验的 FSDP checkpoint 转换为 HuggingFace safetensors 格式
# 合并 FSDP 分片 + 融合 LoRA（--merge_lora），输出完整 dense 模型
#
# 输出目录：/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research/hf_models_<exp>/
#
# 用法：
#   bash tools/convert_all_checkpoints.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_checkpoint.py"
MOUNT="/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research"

EXPERIMENTS=(
    "checkpoints_baseline_qwen3"
    "checkpoints_phase2_alpha0.9_gamma2.0"
    "checkpoints_phase3_filter_only_alpha0.9_gamma0.0"
    "checkpoints_phase4_alpha0.9_gamma0.5"
    "checkpoints_phase4_alpha0.9_gamma1.0"
)

echo "======================================"
echo "共 ${#EXPERIMENTS[@]} 组实验，每组转换所有 global_step_*"
echo "======================================"

for EXP in "${EXPERIMENTS[@]}"; do
    CKPT_DIR="${MOUNT}/${EXP}"
    OUT_DIR="${MOUNT}/hf_models_${EXP#checkpoints_}"   # 去掉前缀 "checkpoints_"

    echo ""
    echo ">>> 实验: ${EXP}"
    echo "    输入: ${CKPT_DIR}"
    echo "    输出: ${OUT_DIR}"

    if [ ! -d "${CKPT_DIR}" ]; then
        echo "    [SKIP] 目录不存在，跳过"
        continue
    fi

    python3 "${CONVERT_SCRIPT}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${OUT_DIR}" \
        --all_steps \
        --merge_lora

    echo "    [DONE] ${EXP}"
done

echo ""
echo "======================================"
echo "全部转换完成"
echo "======================================"
