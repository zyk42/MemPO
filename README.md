# MemPO: Memory-Augmented Policy Optimization

**Mitigating Catastrophic Forgetting in GRPO via Cross-Step EMA Normalization and Mastery Filtering**

## Overview

MemPO (Memory-augmented Policy Optimization, internally developed as EMA-GRPO) addresses a critical failure mode in Group Relative Policy Optimization (GRPO): **catastrophic forgetting of mastered problems during extended reinforcement learning training**.

### Core Idea

Standard GRPO normalizes advantages using per-batch statistics, leading to:
1. High-variance advantage estimates (especially with binary 0/1 rewards)
2. Destructive gradient updates on already-mastered problems
3. Insufficient learning signal on hard problems

MemPO introduces a lightweight **PromptMemoryBank** that maintains per-prompt exponential moving average (EMA) statistics across training steps, with three key innovations:

- **Parameter-free Dynamic Alpha**: `α_t(p) = 1/n + (n-2)/n · (1 - |μ̂_t - μ_{t-1}|)` — automatically adapts EMA smoothing based on policy change speed
- **Mastery Filter**: Zero-gradient protection when EMA mean exceeds τ = (n-1)/n, preventing harmful updates on mastered content
- **No Std Normalization**: Following Dr. GRPO, advantage = r - μ_EMA (no division by std), reducing baseline variance ~10x

## Results

On MATH-500 with Qwen3-1.7B:
- **+2.87% pass@1** over standard GRPO
- **Forgetting collapse entirely eliminated** on GSM8K (+0.79% pass@1)
- Consistent improvements across Qwen2.5-Math-7B and Qwen3-8B

## Repository Structure

```
verl/                                    # Modified verl framework source (for reproduction)
├── verl/trainer/ppo/core_algos.py       # EMA_GRPO advantage estimator
├── verl/trainer/ppo/ray_trainer.py      # Memory bank lifecycle & pre-init
├── verl/trainer/config/algorithm.py     # AlgoConfig with EMA-GRPO fields
├── verl/trainer/config/ppo_trainer.yaml # Default YAML config
├── verl/trainer/main_ppo.py             # Training entry point
├── verl/utils/prompt_memory_bank.py     # Core: per-prompt EMA statistics bank
├── verl/experimental/dataset/
│   ├── sampler.py                       # AbstractCurriculumSampler interface
│   └── degradation_sampler.py           # Degradation-aware replay sampling
└── examples/
    ├── data_preprocess/                 # Dataset preparation scripts
    └── grpo_trainer/                    # Example training scripts

src/                                     # Standalone copies (for quick reference)
├── prompt_memory_bank.py
└── degradation_sampler.py

tools/                                   # Utilities
├── precompute_memory_bank.py            # Pre-init memory bank with base model
├── eval_passk_final.py                  # Pass@k evaluation with vLLM
├── compare_experiments.py               # Cross-experiment comparison
├── merge_lora_direct.py                 # LoRA merge utility
└── ...

run_*.sh                                 # Training scripts (various models/configs)
eval_*.sh                                # Evaluation scripts
eval_results/                            # Experiment results (JSON)
reports/                                 # Detailed experiment reports
fig_data/                                # Training curves and visualization data
paper.tex                                # Paper draft
```

## Reproduction

### Prerequisites

1. Install [verl](https://github.com/volcengine/verl) framework
2. Replace/patch the files in `verl/` directory into your verl installation:
   ```bash
   # Option A: Copy modified files over your verl installation
   cp -r verl/verl/* /path/to/your/verl/verl/
   
   # Option B: Or simply use this repo as the verl source
   cd verl && pip install -e .
   ```
3. Prepare dataset with `index` field in `extra_info`:
   ```python
   # Each row's extra_info must contain: {"index": <int>, "split": "train"}
   python verl/examples/data_preprocess/math_dataset.py
   ```

### Quick Start

MemPO is implemented as a modification to the [verl](https://github.com/volcengine/verl) framework. The complete modified source is in the `verl/` directory.

See [verl_modifications_current.md](verl_modifications_current.md) for detailed integration documentation.

### Training

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ema_grpo \
    algorithm.pre_init_memory_bank=true \
    actor_rollout_ref.rollout.n=8 \
    ...
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_n` | 8 | Determines dynamic α range [1/n, (n-1)/n] and τ = (n-1)/n |
| `pre_init_memory_bank` | true | No-grad rollout init before training |
| `ema_warmup_steps` | 1 | Safety fallback for cold-start |
| `kl_gamma` | 0.0 | Reactive KL penalty (disabled by default) |
| `mastery_soft_alpha` | 0.0 | Soft mastery weighting (disabled by default) |
| `degradation_gamma` | 0.0 | Degradation-aware resampling (disabled by default) |

## Method Details

### Dynamic Alpha (Parameter-Free)

The EMA smoothing factor adapts automatically based on how much the policy has changed:

| Scenario | δ = |μ̂_t - μ_{t-1}| | α | Interpretation |
|----------|-------------------|---|----------------|
| Large policy shift | δ → 1 | α → 1/n | Trust current batch (baseline is stale) |
| Stable policy | δ → 0 | α → (n-1)/n | Trust history (smooth baseline) |

For n=8: α ∈ [0.125, 0.875], no manual tuning required.

### Mastery Filter

When the updated EMA mean μ_t > τ = (n-1)/n:
- Forces advantage = 0 (zero gradient)
- Prevents wasteful/harmful updates on already-solved problems
- Uses μ_t (not batch mean) to avoid false triggers from lucky batches

### Pre-initialization

Before training, runs a no-grad rollout pass over all training prompts to initialize μ₀, eliminating cold-start artifacts.

## Citation

```bibtex
@article{mempo2025,
  title={Mitigating Catastrophic Forgetting in GRPO via Cross-Step EMA Normalization and Mastery Filtering},
  author={Anonymous},
  year={2025}
}
```

## License

Apache License 2.0
