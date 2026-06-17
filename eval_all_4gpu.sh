#!/usr/bin/env bash
# 统一Eval：Qwen3-1.7B × MATH-500 所有方法
# datasets: aime2024 / aime2025 / amc / math500
# n=16, k=[1,8,16], 4 GPU并行
set -e

EXPR_DIR="$HOME/verl/experiments/grpo-forgetting-research"
TOOLS_DIR="$EXPR_DIR/tools"
MNT_DIR="/mnt/lisiqi23/grpo-forgetting-research"
MERGED_DIR="$MNT_DIR/hf_models_qwen3_1.7b_math500"
RESULTS_DIR="$MNT_DIR/eval_results_qwen3_1.7b_math500"
mkdir -p "$RESULTS_DIR"

# 所有需要测试的 (method, step)
ALL_JOBS=(
    "grpo 150"
    "grpo 225"
    "emagrpo 150"
    "emagrpo 225"
    "drgrpo 150"
    "drgrpo 225"
    "emagrpo_dynalpha 150"
    "emagrpo_dynalpha 225"
)

# ── Convert ──
echo "=== Converting checkpoints ==="
for job in "${ALL_JOBS[@]}"; do
    method=$(echo $job | awk '{print $1}')
    step=$(echo $job | awk '{print $2}')
    hf_dir="$MERGED_DIR/${method}_step${step}/global_step_${step}/hf"
    if [ -d "$hf_dir" ]; then
        echo ">>> Skip ${method}_step${step} (exists)"
        continue
    fi
    ckpt_dir="$MNT_DIR/checkpoints_qwen3_1.7b_math500_${method}/global_step_${step}/actor"
    echo ">>> Converting ${method}_step${step} ..."
    CUDA_VISIBLE_DEVICES="" python3 "$TOOLS_DIR/convert_checkpoint.py" \
        --checkpoint_dir "$ckpt_dir" \
        --output_dir "$MERGED_DIR/${method}_step${step}" \
        --merge_lora --lora_alpha 32
done

# ── Eval 4 GPU 并行，每次最多 4 job ──
echo "=== Running evals (4 GPUs) ==="
eval_one() {
    local gpu=$1
    local method=$2
    local step=$3
    local hf_dir="$MERGED_DIR/${method}_step${step}/global_step_${step}/hf"
    local out_file="$RESULTS_DIR/${method}_step${step}.json"
    if [ -f "$out_file" ]; then
        echo ">>> GPU $gpu  SKIP ${method}_step${step} (exists)"
        return 0
    fi
    echo ">>> GPU $gpu  eval ${method}_step${step} ..."
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$TOOLS_DIR/eval_passk_final.py" \
        --model_dir "$hf_dir" \
        --output "$out_file" \
        --n 16 --k 1 8 16 \
        --temperature 0.6 \
        --max_tokens 12288 \
        --tp 1 --gpu_mem 0.80 \
        --datasets aime2024 aime2025 amc math500
    echo ">>> GPU $gpu  done: ${method}_step${step}"
}

# ── 分两批：每批4个job，等一批完成再启动下一批 ──
# 批次1: GPU 0,1,2,3
echo "=== Batch 1/2: GPU 0-3 ==="
for i in 0 1 2 3; do
    job="${ALL_JOBS[$i]}"
    method=$(echo $job | awk '{print $1}')
    step=$(echo $job | awk '{print $2}')
    eval_one $i $method $step &
    sleep 5
done
wait
echo "=== Batch 1 done ==="
sleep 10

# 批次2: GPU 0,1,2,3
echo "=== Batch 2/2: GPU 0-3 ==="
for i in 4 5 6 7; do
    job="${ALL_JOBS[$i]}"
    method=$(echo $job | awk '{print $1}')
    step=$(echo $job | awk '{print $2}')
    gpu=$((i - 4))
    eval_one $gpu $method $step &
    sleep 5
done
wait
echo "=== Batch 2 done ==="

echo ""
echo "=== All Results ==="
python3 - <<'EOF'
import json, glob, os

results_dir = "/mnt/lisiqi23/grpo-forgetting-research/eval_results_qwen3_1.7b_math500"
files = sorted(glob.glob(f"{results_dir}/*.json"))

DATASETS = ["aime2024", "aime2025", "amc", "math500"]
KS = [1, 8, 16]

hdr = f"{'Model':<28}" + "".join(f"  {ds}/p@{k:<3}" for ds in DATASETS for k in KS)
print(hdr)
print("-" * len(hdr))

for fpath in files:
    with open(fpath) as f:
        data = json.load(f)
    name = os.path.basename(fpath).replace(".json", "")
    row = f"{name:<28}"
    for ds in DATASETS:
        for k in KS:
            v = data.get("datasets", {}).get(ds, {}).get("pass_at_k", {}).get(f"pass@{k}", float("nan"))
            row += f"  {v:.3f}  "
    print(row)
EOF
