#!/usr/bin/env bash
# 并行Eval：4 GPU × 4 方法（各 step225），数据集 aime2024/aime2025/amc/math500
set -e

TOOLS="$HOME/verl/experiments/grpo-forgetting-research/tools"
MNT="/mnt/lisiqi23/grpo-forgetting-research"
MERGED="$MNT/hf_models_qwen3_1.7b_math500"
RESULTS="$MNT/eval_results_qwen3_1.7b_math500"
mkdir -p "$RESULTS"

# 4 个任务: GPU, method, step
JOBS=(
    "0 grpo 225"
    "1 emagrpo 225"
    "2 drgrpo 225"
    "3 emagrpo_dynalpha 225"
)

eval_one() {
    local gpu=$1; shift
    local method=$1; shift
    local step=$1
    local hf_dir="$MERGED/${method}_step${step}/global_step_${step}/hf"
    local out="$RESULTS/${method}_step${step}.json"
    echo ">>> GPU $gpu  ${method}_step${step}  (datasets: aime2024 aime2025 amc math500)"
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$TOOLS/eval_passk_final.py" \
        --model_dir "$hf_dir" \
        --output "$out" \
        --n 16 --k 1 8 16 \
        --temperature 0.6 \
        --max_tokens 12288 \
        --tp 1 --gpu_mem 0.6 \
        --datasets aime2024 aime2025 amc math500
    echo ">>> GPU $gpu  done: ${method}_step${step}"
}

for job in "${JOBS[@]}"; do
    gpu=$(echo $job | awk '{print $1}')
    method=$(echo $job | awk '{print $2}')
    step=$(echo $job | awk '{print $3}')
    eval_one $gpu $method $step &
done
wait

echo ""
echo "=== Results ==="
python3 - <<'EOF'
import json, glob, os

results_dir = "/mnt/lisiqi23/grpo-forgetting-research/eval_results_qwen3_1.7b_math500"
DATASETS = ["aime2024", "aime2025", "amc", "math500"]
KS = [1, 8, 16]

hdr = f"{'Model':<24}" + "".join(f"  {ds}/p@{k}" for ds in DATASETS for k in KS)
print(hdr)
print("-" * len(hdr))

for fpath in sorted(glob.glob(f"{results_dir}/*.json")):
    with open(fpath) as f:
        data = json.load(f)
    name = os.path.basename(fpath).replace(".json", "")
    row = f"{name:<24}"
    for ds in DATASETS:
        for k in KS:
            v = data.get("datasets",{}).get(ds,{}).get("pass_at_k",{}).get(f"pass@{k}", float("nan"))
            row += f"  {v:.3f}"
    print(row)
EOF
