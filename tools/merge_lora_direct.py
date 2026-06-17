#!/usr/bin/env python3
"""直接从 verl FSDP checkpoint 合并 LoRA 到 base 权重，保存为 safetensors。

绕过 PEFT 的 lora_alpha 读取（verl 保存的 adapter_config 中 lora_alpha=0，实为 bug），
手动按训练时的 lora_alpha/r 计算 scaling 并合并。

用法：
  python tools/merge_lora_direct.py \\
      --ckpt_dir /mnt/.../checkpoints_baseline_qwen3/global_step_480/actor \\
      --output_dir /mnt/.../hf_models_baseline_qwen3/global_step_480/hf \\
      --lora_alpha 32 --lora_r 64

  # 批量转换所有实验所有 step
  python tools/merge_lora_direct.py --batch --lora_alpha 32 --lora_r 64
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.tensor  # 必须导入才能 unpickle DTensor


MOUNT = Path("/mnt/lisiqi23/internship/zhangyikang/grpo-forgetting-research")

EXPERIMENTS = [
    ("checkpoints_baseline_qwen3",                        "hf_models_baseline_qwen3"),
    ("checkpoints_phase2_alpha0.9_gamma2.0",              "hf_models_phase2_alpha0.9_gamma2.0"),
    ("checkpoints_phase3_filter_only_alpha0.9_gamma0.0",  "hf_models_phase3_filter_only_alpha0.9_gamma0.0"),
    ("checkpoints_phase4_alpha0.9_gamma0.5",              "hf_models_phase4_alpha0.9_gamma0.5"),
    ("checkpoints_phase4_alpha0.9_gamma1.0",              "hf_models_phase4_alpha0.9_gamma1.0"),
    ("checkpoints_phase5_soft_alpha0.9_soft1.0",          "hf_models_phase5_soft_alpha0.9_soft1.0"),
    ("checkpoints_phase5_soft_alpha0.9_soft0.5",          "hf_models_phase5_soft_alpha0.9_soft0.5"),
    ("checkpoints_phase5_soft_alpha0.9_soft2.0",          "hf_models_phase5_soft_alpha0.9_soft2.0"),
    ("checkpoints_phase6_reactive_alpha0.9_gamma0.5",      "hf_models_phase6_reactive_alpha0.9_gamma0.5"),
    ("checkpoints_phase6_reactive_alpha0.9_gamma1.0",      "hf_models_phase6_reactive_alpha0.9_gamma1.0"),
    ("checkpoints_phase6_reactive_alpha0.9_gamma2.0",      "hf_models_phase6_reactive_alpha0.9_gamma2.0"),
    ("checkpoints_phase3_filter_only_alpha0.9_gamma0.0",   "hf_models_phase3_filter_only_alpha0.9_gamma0.0"),
]


# ---------------------------------------------------------------------------
# FSDP 分片加载与合并
# ---------------------------------------------------------------------------

def load_fsdp_shards(actor_dir: Path) -> dict[str, torch.Tensor]:
    """加载所有 model_world_size_N_rank_K.pt，拼接 DTensor 分片为完整 tensor。"""
    import re
    pt_files = sorted(
        actor_dir.glob("model_world_size_*_rank_*.pt"),
        key=lambda p: int(re.search(r"rank_(\d+)", p.name).group(1)),
    )
    if not pt_files:
        raise FileNotFoundError(f"No model pt files in {actor_dir}")

    print(f"  [fsdp] 加载 {len(pt_files)} 个分片 ...")
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in pt_files]

    merged = {}
    for key in shards[0].keys():
        tensors = [s[key] for s in shards]
        # DTensor: 用 to_local() 取本地分片后拼接
        if hasattr(tensors[0], "to_local"):
            local_shards = [t.to_local() for t in tensors]
            placements = tensors[0].placements
            # Shard(0): 沿 dim 0 拼接; Replicate: 任取一个
            from torch.distributed.tensor import Shard, Replicate
            if any(isinstance(p, Shard) for p in placements):
                shard_dim = next(p.dim for p in placements if isinstance(p, Shard))
                merged[key] = torch.cat(local_shards, dim=shard_dim)
            else:
                merged[key] = local_shards[0]
        else:
            merged[key] = tensors[0]

    print(f"  [fsdp] 合并完成，共 {len(merged)} 个参数")
    return merged


# ---------------------------------------------------------------------------
# LoRA 合并
# ---------------------------------------------------------------------------

def merge_lora(state_dict: dict[str, torch.Tensor], lora_alpha: float, lora_r: int) -> dict[str, torch.Tensor]:
    """将 lora_A / lora_B 融合进 base_layer，返回标准 HF 格式的 state_dict。

    PEFT 命名：base_model.model.{hf_name}.base_layer.weight
               base_model.model.{hf_name}.lora_A.default.weight
               base_model.model.{hf_name}.lora_B.default.weight
    HF 命名：  {hf_name}.weight
    """
    scaling = lora_alpha / lora_r
    print(f"  [lora] scaling = {lora_alpha}/{lora_r} = {scaling}")

    # 找出所有有 lora 的层
    lora_a_keys = {k for k in state_dict if "lora_A.default.weight" in k}
    merged_layers = set()

    result = {}

    for key, tensor in state_dict.items():
        tensor = tensor.float()

        if "base_layer.weight" in key:
            # base_model.model.X.base_layer.weight → model.X.weight
            hf_key = key.replace("base_model.model.", "").replace(".base_layer.weight", ".weight")
            # 找对应的 lora_A / lora_B
            prefix = key.replace(".base_layer.weight", "")
            lora_a_key = f"{prefix}.lora_A.default.weight"
            lora_b_key = f"{prefix}.lora_B.default.weight"

            if lora_a_key in state_dict and lora_b_key in state_dict:
                lora_a = state_dict[lora_a_key].float()
                lora_b = state_dict[lora_b_key].float()
                delta = lora_b @ lora_a * scaling
                result[hf_key] = (tensor + delta).to(torch.bfloat16)
                merged_layers.add(prefix)
            else:
                result[hf_key] = tensor.to(torch.bfloat16)

        elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
            # 跳过，已经融入 base_layer
            continue

        elif key.startswith("base_model.model."):
            # 其他非 lora 层（如 norm、embed 等）
            hf_key = key.replace("base_model.model.", "")
            result[hf_key] = tensor.to(torch.bfloat16)

        else:
            result[key] = tensor.to(torch.bfloat16)

    print(f"  [lora] 融合了 {len(merged_layers)} 个 LoRA 层，输出 {len(result)} 个参数")
    return result


# ---------------------------------------------------------------------------
# 保存为 HF safetensors
# ---------------------------------------------------------------------------

def save_hf_model(state_dict: dict[str, torch.Tensor], hf_src_dir: Path, output_dir: Path):
    """保存 state_dict 为 safetensors，并从 hf_src_dir 复制 tokenizer/config 文件。"""
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [save] 保存 model.safetensors 到 {output_dir} ...")
    save_file(state_dict, output_dir / "model.safetensors")

    # 复制 tokenizer 和 config 文件
    for f in hf_src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)

    print(f"  [save] 完成，文件大小: {(output_dir / 'model.safetensors').stat().st_size / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# 单个 checkpoint 转换
# ---------------------------------------------------------------------------

def convert_one(actor_dir: Path, output_dir: Path, lora_alpha: float, lora_r: int):
    print(f"\n{'='*60}")
    print(f"  输入: {actor_dir}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}")

    if (output_dir / "model.safetensors").exists():
        print("  [SKIP] 已存在，跳过")
        return

    state_dict = load_fsdp_shards(actor_dir)
    merged = merge_lora(state_dict, lora_alpha, lora_r)
    hf_src = actor_dir / "huggingface"
    save_hf_model(merged, hf_src, output_dir)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="批量转换所有实验")
    parser.add_argument("--ckpt_dir", default=None, help="单个 actor 目录")
    parser.add_argument("--output_dir", default=None, help="单个输出目录")
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_r", type=int, default=64)
    args = parser.parse_args()

    if args.batch:
        for ckpt_name, hf_name in EXPERIMENTS:
            ckpt_root = MOUNT / ckpt_name
            hf_root = MOUNT / hf_name
            for step_dir in sorted(ckpt_root.glob("global_step_*"),
                                   key=lambda p: int(p.name.split("_")[-1])):
                actor_dir = step_dir / "actor"
                output_dir = hf_root / step_dir.name / "hf"
                convert_one(actor_dir, output_dir, args.lora_alpha, args.lora_r)
        print("\n全部完成。")
    else:
        if not args.ckpt_dir or not args.output_dir:
            parser.error("非 --batch 模式需要指定 --ckpt_dir 和 --output_dir")
        convert_one(Path(args.ckpt_dir), Path(args.output_dir), args.lora_alpha, args.lora_r)


if __name__ == "__main__":
    main()
