#!/usr/bin/env python3
"""将 verl FSDP 分片 checkpoint 转换为标准 HuggingFace safetensors 格式。

verl 训练时以 FSDP 分片形式保存 checkpoint：
  checkpoints/global_step_240/actor/
      model_world_size_4_rank_0.pt
      model_world_size_4_rank_1.pt
      ...
      huggingface/          ← 仅含 tokenizer/config，无模型权重

本脚本完成以下步骤：
  1. 调用 verl 内置 legacy_model_merger 合并 FSDP 分片
  2. （可选）将 LoRA adapter 合并进 base model，得到完整的 dense 模型
  3. 以 safetensors 格式保存到目标目录

用法示例：
  # 仅合并 FSDP 分片，保留 LoRA adapter 格式
  python tools/convert_checkpoint.py \\
      --checkpoint_dir checkpoints_alpha0.9_warmup3/global_step_240/actor \\
      --output_dir ./merged_step240

  # 合并 FSDP 分片 + 融合 LoRA 为 dense 模型
  python tools/convert_checkpoint.py \\
      --checkpoint_dir checkpoints_alpha0.9_warmup3/global_step_240/actor \\
      --output_dir ./merged_step240_dense \\
      --merge_lora

  # 转换多个 checkpoint（批量）
  python tools/convert_checkpoint.py \\
      --checkpoint_dir checkpoints_alpha0.9_warmup3 \\
      --output_dir ./merged \\
      --all_steps \\
      --merge_lora
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


VERL_ROOT = Path(__file__).resolve().parents[3]  # verl/
MERGER_SCRIPT = VERL_ROOT / "scripts" / "legacy_model_merger.py"


def find_actor_dirs(root: Path) -> list[Path]:
    """在 root 下找到所有 global_step_*/actor 目录，按步数排序。"""
    actor_dirs = sorted(
        root.glob("global_step_*/actor"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
    )
    return actor_dirs


def merge_fsdp(actor_dir: Path, output_dir: Path) -> Path:
    """调用 legacy_model_merger 合并 FSDP 分片到 HuggingFace 格式。

    输出目录：output_dir/hf/
    """
    hf_out = output_dir / "hf"
    hf_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(MERGER_SCRIPT),
        "merge",
        "--backend", "fsdp",
        "--local_dir", str(actor_dir),
        "--target_dir", str(hf_out),
    ]
    print(f"[merge_fsdp] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return hf_out


def fix_lora_alpha(adapter_dir: Path, lora_alpha: int):
    """Patch adapter_config.json to set the correct lora_alpha.

    legacy_model_merger hardcodes lora_alpha=0 in adapter_config.json,
    which causes LoRA scaling = lora_alpha/r = 0, zeroing all LoRA updates.
    This must be corrected before merging.
    """
    import json
    cfg_path = adapter_dir / "adapter_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    old_alpha = cfg.get("lora_alpha", 0)
    cfg["lora_alpha"] = lora_alpha
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"[fix_lora_alpha] lora_alpha: {old_alpha} -> {lora_alpha}")


def merge_lora_into_base(hf_dir: Path, adapter_dir: Path, output_dir: Path, lora_alpha: int) -> Path:
    """将 LoRA adapter 融合进 base model，保存为 dense safetensors。

    hf_dir:     包含 config.json / model.safetensors / tokenizer 的基础目录
    adapter_dir: 包含 adapter_config.json / adapter_model.safetensors 的目录
    lora_alpha: 训练时使用的 lora_alpha 值（legacy_model_merger 将其硬编码为 0，需手动修正）
    需要 peft 库。
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dense_out = output_dir / "dense"
    dense_out.mkdir(parents=True, exist_ok=True)

    # Fix lora_alpha before loading adapter (legacy_model_merger hardcodes 0)
    fix_lora_alpha(adapter_dir, lora_alpha)

    print(f"[merge_lora] Loading base model from {hf_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="cpu",
    )
    print(f"[merge_lora] Loading LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("[merge_lora] Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"[merge_lora] Saving dense model to {dense_out} ...")
    model.save_pretrained(dense_out, safe_serialization=True)
    tokenizer.save_pretrained(dense_out)
    print(f"[merge_lora] Done: {dense_out}")
    return dense_out


def is_lora_checkpoint(hf_dir: Path) -> bool:
    """判断 hf_dir 是否包含 LoRA adapter。
    legacy_model_merger 将 adapter 保存在 hf_dir/lora_adapter/ 子目录下。
    """
    return (hf_dir / "adapter_config.json").exists() or \
           (hf_dir / "lora_adapter" / "adapter_config.json").exists()


def get_lora_adapter_dir(hf_dir: Path) -> Path:
    """返回 LoRA adapter 目录（adapter_config.json 所在位置）。"""
    if (hf_dir / "adapter_config.json").exists():
        return hf_dir
    return hf_dir / "lora_adapter"


def convert_one(actor_dir: Path, output_dir: Path, merge_lora: bool, lora_alpha: int = 32):
    step_name = actor_dir.parent.name  # global_step_240
    step_out = output_dir / step_name
    step_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Converting: {actor_dir}")
    print(f"Output:     {step_out}")
    print(f"{'='*60}")

    # Step 1: 合并 FSDP 分片
    hf_dir = merge_fsdp(actor_dir, step_out)

    # Step 2: 可选融合 LoRA
    if merge_lora:
        if is_lora_checkpoint(hf_dir):
            adapter_dir = get_lora_adapter_dir(hf_dir)
            print(f"[convert] Found LoRA adapter at {adapter_dir}")
            final_dir = merge_lora_into_base(hf_dir, adapter_dir, step_out, lora_alpha=lora_alpha)
            # 删除中间 hf/ 目录节省空间
            shutil.rmtree(hf_dir)
            final_dir.rename(hf_dir)
            print(f"[convert] Replaced hf/ with dense model")
        else:
            print(f"[convert] No LoRA adapter found in {hf_dir}, skipping LoRA merge")

    # 打印最终目录内容
    print(f"\n[convert] Output files:")
    for f in sorted(hf_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
        print(f"  {f.name:<40} {size_mb:6.1f} MB" if f.is_file() else f"  {f.name}/")

    return hf_dir


def main():
    parser = argparse.ArgumentParser(description="Convert verl FSDP checkpoint to HuggingFace safetensors")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="actor 目录路径（如 checkpoints/global_step_240/actor）"
                             "或 checkpoints 根目录（配合 --all_steps 使用）")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录")
    parser.add_argument("--merge_lora", action="store_true",
                        help="将 LoRA adapter 融合进 base model，输出 dense 模型")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="训练时的 lora_alpha 值（legacy_model_merger 将其硬编码为 0，需手动修正）")
    parser.add_argument("--all_steps", action="store_true",
                        help="转换 checkpoint_dir 下所有 global_step_* 子目录")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)

    if args.all_steps:
        actor_dirs = find_actor_dirs(ckpt_dir)
        if not actor_dirs:
            print(f"[ERROR] 未在 {ckpt_dir} 下找到 global_step_*/actor 目录")
            sys.exit(1)
        print(f"找到 {len(actor_dirs)} 个 checkpoint: {[d.parent.name for d in actor_dirs]}")
        for actor_dir in actor_dirs:
            convert_one(actor_dir, out_dir, args.merge_lora, lora_alpha=args.lora_alpha)
    else:
        # checkpoint_dir 直接是 actor 目录
        actor_dir = ckpt_dir
        if not actor_dir.name == "actor":
            # 兼容传入 global_step_N 目录
            actor_dir = ckpt_dir / "actor"
        if not actor_dir.exists():
            print(f"[ERROR] 目录不存在: {actor_dir}")
            sys.exit(1)
        convert_one(actor_dir, out_dir, args.merge_lora)

    print("\n[convert] 全部完成。")


if __name__ == "__main__":
    main()
