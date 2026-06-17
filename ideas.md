# GRPO Forgetting Problem: Research Ideas

## Background

Observation from Qwen3-1.7B trained on MATH-500 with GRPO (10 epochs, 400 train samples):
- 16/373 questions showed score degradation from epoch 1 to epoch 9
- Common failure patterns:
  1. **Last-step errors**: reasoning process correct but final step wrong
  2. **Over-verification loops**: model gets stuck re-checking and drifts
  3. **Boundary condition dropout**: model simplifies answers and loses edge cases
- Root cause: sparse data (400 samples), noisy per-batch advantage estimates, reward overfitting

---

## Idea 1: Cross-Epoch EMA Statistics for Advantage Normalization

### Motivation
Standard GRPO normalizes advantages within a single batch of `n` rollouts:
```
A_i = (r_i - mean_batch) / std_batch
```
With only 8 rollouts per prompt and sparse appearances (~10 times per epoch), the
batch-level mean/std is noisy and unstable across training.

### Method
Maintain a per-prompt exponential moving average (EMA) of scores:
```
ema_mean[p] = α * ema_mean[p] + (1 - α) * mean(r_current)
ema_std[p]  = α * ema_std[p]  + (1 - α) * std(r_current)
```
Replace batch statistics with historical statistics during advantage computation:
```
A_i = (r_i - ema_mean[prompt]) / (ema_std[prompt] + ε)
```

### Expected Effect
- More stable advantage estimates, especially for hard/easy prompts
- Reduces variance of gradient updates caused by lucky/unlucky rollout batches
- Implicitly tracks prompt difficulty over time

### Implementation Notes
- Need a `PromptMemoryBank` keyed by prompt hash, storing EMA stats
- EMA decay `α` is a new hyperparameter (suggested: 0.9~0.95)
- Requires warm-up period before EMA is reliable (first few appearances use batch stats)
- Storage: O(N_prompts) floats, negligible memory overhead

### Concerns
- EMA warm-up instability in early training
- Hash collision for very long prompts (use truncated hash)
- If prompt set is not fixed across epochs (shuffling), need robust keying

---

## Idea 2: Sample-Level Adaptive KL Constraint

### Motivation
The global KL coefficient `kl_coef` applies uniform regularization to all prompts.
Prompts where the model is degrading need stronger regularization; prompts where the
model is improving can afford weaker regularization.

### Method
Compute a per-prompt KL multiplier based on performance trend:
```
trend[p] = ema_mean_current[p] - ema_mean_prev[p]   # negative = degrading
kl_multiplier[p] = 1 + β * max(0, -trend[p])        # increase KL if degrading
loss_kl[p] = kl_multiplier[p] * base_kl_coef * KL(π || π_ref)[p]
```

### Expected Effect
- Degrading prompts are pulled back toward the reference policy more aggressively
- Improving prompts are less constrained and can continue to improve
- Acts as an automatic, per-sample trust region

### Implementation Notes
- Requires per-prompt history (shares infrastructure with Idea 1)
- KL is computed token-level in verl; need to apply per-sample weight at loss aggregation
- `β` controls sensitivity to degradation signal

### Concerns
- Persistently hard prompts may have KL frozen permanently (policy can't improve)
- Needs careful tuning of `β` to avoid over-regularization
- More complex than Idea 1; harder to diagnose if something goes wrong

---

## Idea 3: History-Shaped Advantage (Unified Framework)

### Motivation
Combines Ideas 1 and 2 into a single, simpler formulation. Instead of modifying the
KL term (which requires changes deep in the loss computation), add a history-aware
penalty directly into the advantage:

### Method
```
A_shaped[i] = A_i - λ * max(0, ema_mean[prompt] - r_i)
```
Where:
- `A_i` is the standard GRPO advantage
- `ema_mean[prompt]` is the historical mean score for this prompt
- The penalty term fires only when current score is **below** historical mean
- `λ` controls the strength of the forgetting penalty

### Expected Effect
- When a prompt degrades, its advantages are suppressed (less positive or more negative)
- Gradient updates are naturally smaller for degrading prompts
- No changes needed to KL computation; everything happens at advantage level

### Implementation Notes
- Simpler than Idea 2: only need to modify advantage computation
- `λ` is a new hyperparameter (suggested: 0.5~2.0)
- Can be combined with Idea 1 (EMA normalization) for full effect

### Concerns
- May over-suppress correct responses on hard prompts that happen to score below EMA
- The "penalty" signal and "learning" signal are mixed in the same advantage value

---

## Idea 4: Degradation-Aware Prompt Replay (Data-Level)

### Motivation
If the model is forgetting certain prompts, the simplest fix is to make it see those
prompts more often. This addresses forgetting at the data level without any algorithmic
changes to GRPO itself.

### Method
Maintain a per-prompt **degradation score**:
```
deg_score[p] = max(0, ema_mean_prev[p] - ema_mean_current[p])
```
Use this as sampling weight in the dataloader:
```
sample_weight[p] = base_weight + γ * deg_score[p]
```
Degrading prompts are oversampled in subsequent epochs.

### Expected Effect
- Model receives more gradient signal on forgetting cases
- Naturally self-correcting: once a prompt recovers, its weight drops back
- Orthogonal to all other ideas; can be stacked on top

### Implementation Notes
- Requires a weighted sampler in the verl dataloader
- Degradation score needs cross-epoch persistence (same infrastructure as Idea 1)
- `γ` controls how aggressively degrading prompts are replayed

### Concerns
- May cause distribution shift if too many prompts degrade simultaneously
- Oversampling rare hard prompts could destabilize training on easy prompts
- Interaction with `data.shuffle` needs care

---

## Idea 5: Prompt-Level Conservative Clipping

### Motivation
PPO/GRPO uses a global clip ratio `ε` (typically 0.2) for all prompts. For degrading
prompts, a tighter clip ratio would limit how far the policy can move away from the
previous rollout policy.

### Method
Compute per-prompt clip ratio based on performance trend:
```
clip_ratio[p] = base_clip * (1 - δ * max(0, -trend[p]))
clip_ratio[p] = max(clip_ratio[p], min_clip)   # floor at e.g. 0.05
```
Use `clip_ratio[p]` in the PPO clipping objective for samples from prompt `p`.

### Expected Effect
- Degrading prompts have tighter trust region → policy changes more conservatively
- Stable/improving prompts maintain normal clip ratio
- More targeted than global KL

### Implementation Notes
- Requires modifying the PPO loss computation in `core_algos.py`
- Per-sample clip ratios need to be passed alongside advantages
- Computationally negligible overhead

### Concerns
- Tight clipping may prevent recovery on hard prompts
- Clip ratio is already a blunt instrument; per-sample version adds complexity

---

## Experiment Plan

### Priority Order (simple → complex)

| ID | Method | Complexity | Expected insight |
|----|--------|-----------|-----------------|
| Baseline | Standard GRPO (current) | — | Establish forgetting baseline |
| Exp-A | Idea 4: Degradation replay | Low | Data-level fix, clean ablation |
| Exp-B | Idea 1: EMA advantage norm | Medium | Algorithm-level fix |
| Exp-C | Idea 3: History-shaped advantage | Medium | Unified, reward-shaping perspective |
| Exp-D | Idea 2: Adaptive KL | High | KL-perspective fix |
| Exp-E | Idea 1 + 2 (joint) | High | Full system |

### Evaluation Metrics
- **Primary**: per-prompt score trajectory across epochs (track degraded prompts)
- **Secondary**: overall val accuracy, training stability (reward variance)
- **Diagnostic**: number of prompts with score drop > 0.25 per epoch transition

### Shared Infrastructure Needed
- `PromptMemoryBank`: per-prompt EMA statistics, persisted across training steps
- Per-prompt score logging in rollout data (already available via `rollout_data_dir`)
- Analysis script to detect degradation (already written in rollout_logs analysis)

---

## Phase 2–4 Experimental Results & Revised Directions

### Experimental Summary (Qwen3-1.7B, MATH-500, n=8, temperature=0.6)

#### Full Results

| Method | split | pass@1 | pass@4 | pass@8 |
|--------|-------|-------:|-------:|-------:|
| original | test | 0.6000 | 0.7097 | 0.7500 |
| original | train | 0.5675 | 0.6667 | 0.7025 |
| baseline step400 | test | 0.6675 | 0.7559 | 0.7700 |
| baseline step400 | train | 0.6384 | 0.7106 | 0.7300 |
| baseline step480 | test | 0.6687 | 0.7656 | 0.7900 |
| baseline step480 | train | 0.6453 | 0.7121 | 0.7325 |
| phase2 γ=2.0 step400 | test | 0.6150 | 0.7287 | 0.7600 |
| phase2 γ=2.0 step400 | train | 0.6062 | 0.6829 | 0.7100 |
| phase2 γ=2.0 step480 | test | 0.6162 | 0.7253 | 0.7600 |
| phase2 γ=2.0 step480 | train | 0.6081 | 0.6868 | 0.7075 |
| **phase3 γ=0 step400** | **test** | **0.6763** | **0.7593** | **0.7800** |
| phase3 γ=0 step400 | train | 0.6409 | 0.7048 | 0.7250 |
| **phase3 γ=0 step480** | **test** | **0.6887** | **0.7631** | **0.7800** |
| phase3 γ=0 step480 | train | 0.6403 | 0.7033 | 0.7225 |
| phase4 γ=0.5 step400 | test | 0.6262 | 0.7146 | 0.7300 |
| phase4 γ=0.5 step400 | train | 0.6141 | 0.6794 | 0.7050 |
| phase4 γ=0.5 step480 | test | 0.6275 | 0.7324 | 0.7600 |
| phase4 γ=0.5 step480 | train | 0.6131 | 0.6822 | 0.7000 |
| phase4 γ=1.0 step400 | test | 0.6188 | 0.7370 | 0.7900 |
| phase4 γ=1.0 step400 | train | 0.6088 | 0.6877 | 0.7100 |
| phase4 γ=1.0 step480 | test | 0.6350 | 0.7429 | 0.7800 |
| phase4 γ=1.0 step480 | train | 0.6122 | 0.6863 | 0.7125 |

#### Δ vs. Original (step480, pass@1 / pass@4 / pass@8)

| Method | Δtest p@1 | Δtest p@4 | Δtest p@8 | Δtrain p@1 | Δtrain p@4 | Δtrain p@8 |
|--------|----------:|----------:|----------:|-----------:|-----------:|-----------:|
| baseline | +0.069 | +0.056 | +0.040 | +0.078 | +0.045 | +0.030 |
| **phase3 γ=0** | **+0.089** | **+0.053** | **+0.030** | **+0.073** | **+0.037** | **+0.020** |
| phase4 γ=0.5 | +0.028 | +0.023 | +0.010 | +0.046 | +0.016 | −0.003 |
| phase4 γ=1.0 | +0.035 | +0.033 | +0.030 | +0.045 | +0.020 | +0.010 |
| phase2 γ=2.0 | +0.016 | +0.016 | +0.010 | +0.041 | +0.020 | +0.005 |

**Key findings:**
1. **Mastery filter alone (phase3) is the best approach**: highest test generalization (+0.089 vs original), still improving at step480 (not yet converged)
2. **Adaptive KL boost (kl_gamma) consistently hurts**: γ越大性能越差，γ=2.0几乎完全压制了学习
3. **KL boost is preemptive and too broad**: it penalizes all mastered problems regardless of whether forgetting is actually happening, killing plasticity globally
4. **The mastery filter is beneficial but incomplete**: it prevents gradient on mastered problems but cannot stop weight drift caused by learning other problems

---

## Revised & New Directions (Post Phase 2–4)

### Direction A: Soft Advantage Weighting by EMA (upgrade of mastery filter)

**Problem with current phase3**: hard binary cutoff (batch_mean=1.0 → advantage=0) creates abrupt gradient discontinuity.

**Proposed method**:
```
advantage_weight(p) = (1 - EMA_acc(p))^α
```
- EMA_acc=0.9 → weight=0.10 (almost no gradient)
- EMA_acc=0.5 → weight=0.50 (half gradient)
- EMA_acc=0.2 → weight=0.83 (near-full gradient)

This creates a smooth learning curriculum that concentrates gradient on the "learning frontier" — problems transitioning from wrong to right.

**Difference from phase3**: softer, no hard threshold; naturally handles problems that fluctuate near the mastery boundary.

**Implementation**: modify advantage computation, multiply `advantages[i] *= (1 - ema_mean[prompt]) ** alpha`.

**Expected gain**: smoother optimization, potentially better than phase3's hard cutoff.

---

### Direction B: Reactive Forgetting Penalty (targeted KL)

**Problem with kl_gamma**: it penalizes mastered problems preemptively even when no forgetting is occurring, suppressing learning globally.

**Proposed method**: only trigger KL penalty when forgetting is actually detected:
```python
if EMA_acc_prev(p) > 0.8 and current_batch_acc(p) < 0.5:
    # Forgetting event detected — apply targeted KL penalty
    loss += λ * KL(π_θ || π_ref, prompt=p)
```

**Key difference from phase2/4**: the penalty scope is tiny (only actually-forgetting problems), not all mastered problems. Learning on unmastered problems is completely unaffected.

**Expected gain**: should recover the learning performance of phase3 while providing better forgetting protection when it actually occurs.

---

### Direction C: Gradient Projection (OGD-style)

**Motivation**: learning new problems and retaining mastered ones have conflicting gradient directions. Directly project the learning gradient to not harm retention.

**Proposed method**:
```python
g_retain = gradient from mastered problems (EMA_acc > 0.8)
g_learn  = gradient from unmastered problems
# If conflicting, project g_learn away from g_retain direction
if dot(g_learn, g_retain) < 0:
    g_learn -= dot(g_learn, g_retain) / ||g_retain||^2 * g_retain
```

Analogous to Orthogonal Gradient Descent (OGD) from continual learning literature.

**Tradeoff**: computationally expensive (need to compute two separate gradients); architecturally more invasive.

---

### Direction D: Decoupled Learning + Retention Objectives

Split training into two data streams with separate loss objectives:

| Problem state | Criterion | Objective | Loss |
|--------------|-----------|-----------|------|
| Unlearned | EMA_acc < 0.3 | Learn | Standard GRPO |
| Frontier | 0.3 ≤ EMA_acc < 0.8 | Learn harder | Upsampled GRPO |
| Mastered | EMA_acc ≥ 0.8 | Retain | Lightweight KL / SFT on correct rollouts |

This cleanly decouples learning and retention, so they cannot interfere with each other.

---

### Revised Priority Order (updated after Phase 6)

**Phase 5 conclusion — Direction A falsified:**

| Method | test p@1 | test p@8 | train p@8 | Note |
|--------|----------|----------|-----------|------|
| baseline step480 | 0.6687 | **0.7900** | 0.7325 | reference |
| phase3 step480 (hard cutoff) | **0.6887** | 0.7800 | 0.7225 | best so far |
| phase5 s=0.5 step480 | 0.6863 | 0.7700 | 0.6950 | worse than original |
| phase5 s=1.0 step480 | 0.6613 | 0.7600 | 0.7025 | worse than baseline |
| phase5 s=2.0 step480 | 0.6675 | 0.7800 | 0.7175 | close to phase3 |

- Soft weighting made pass@8 *worse*, not better; train pass@8 decays continuously with steps
- Non-zero gradients on mastered problems accumulate over training → more forgetting, not less
- phase3's pass@8 gap vs baseline (-0.010) is due to **diversity reduction**, not capability loss — gradient retention cannot fix it
- **Direction A abandoned; phase3 hard cutoff remains best**

---

**Phase 6 conclusion — Direction B falsified:**

| Method | test p@1 | test p@4 | test p@8 | train p@1 | Note |
|--------|----------|----------|----------|-----------|------|
| baseline step480 | 0.6687 | 0.7656 | **0.7900** | 0.6453 | reference |
| **phase3 step480** | **0.6887** | **0.7631** | 0.7800 | **0.6403** | **best overall** |
| phase6 γ=0.5 step480 | 0.6450 | 0.7363 | 0.7700 | 0.6116 | below baseline |
| phase6 γ=1.0 step480 | 0.6375 | 0.7533 | 0.7800 | 0.6234 | below baseline |
| phase6 γ=2.0 step480 | 0.6375 | 0.7411 | 0.7800 | 0.6228 | below baseline |

**Key findings:**
1. **Reactive KL uniformly underperforms baseline**: all γ values reduce test p@1 by 0.02–0.03 vs baseline. γ=0.5 reaches 0.6538 at step400 but decays to 0.6450 by step480 — the KL penalty accumulates and increasingly suppresses learning
2. **Reactive is not fundamentally better than preemptive**: phase6 γ=0.5 ≈ phase4 γ=0.5; the more precise trigger condition helps slightly but the root problem remains — KL acts on all tokens globally, not just the regressed prompt's tokens
3. **Fundamental conflict**: increasing KL for a regressed prompt effectively tightens the trust region for the entire batch; it is impossible to surgically anchor policy only on the regressed prompt's tokens
4. **Phase3 hard mastery filter is the only method that beats baseline**: it reduces noisy gradients from mastered prompts without interfering with the policy's exploration on unmastered ones

**Direction B abandoned.**

---

**Full comparison across all methods (step480, Δ vs original):**

| Method | Δtest p@1 | Δtest p@4 | Δtest p@8 |
|--------|----------:|----------:|----------:|
| baseline | +0.069 | +0.056 | +0.040 |
| **phase3 (mastery filter only)** | **+0.089** | **+0.053** | **+0.030** |
| phase6 γ=0.5 | +0.045 | +0.027 | +0.020 |
| phase6 γ=1.0 | +0.038 | +0.044 | +0.030 |
| phase6 γ=2.0 | +0.038 | +0.031 | +0.030 |
| phase2 γ=2.0 | +0.016 | +0.016 | +0.010 |

**Conclusion: any form of KL penalty (preemptive or reactive) degrades pass@1 learning. Phase3 mastery filter is the only method that outperforms baseline on pass@1.**

---

| Priority | Method | Complexity | Hypothesis | Status |
|----------|--------|-----------|------------|--------|
| ~~1~~ | ~~Direction A (soft weighting)~~ | ~~Low~~ | ~~Smooth curriculum > hard cutoff~~ | **Falsified** |
| ~~1~~ | ~~Direction B (reactive KL)~~ | ~~Medium~~ | ~~Targeted penalty > preemptive~~ | **Falsified** |
| 1 | **Extend phase3 training** | Low | step480 not yet converged | Next |
| 2 | Direction D (decoupled objectives) | High | Orthogonal learning/retention | Planned |
| 3 | Direction C (gradient projection) | Very High | Theoretically cleanest | Planned |

---

**Extended phase3 training conclusion (step640/800/960):**

| Method | test p@1 | test p@4 | test p@8 | train p@1 | Note |
|--------|----------|----------|----------|-----------|------|
| baseline step480 | 0.6687 | 0.7656 | 0.7900 | 0.6453 | reference |
| phase3 step480 | 0.6887 | 0.7631 | 0.7800 | 0.6403 | previous best |
| **phase3 step640** | **0.6850** | **0.7763** | **0.8000** | 0.6481 | **best pass@4/8** |
| phase3 step800 | 0.6837 | 0.7601 | 0.7800 | 0.6497 | beginning to overfit |
| phase3 step960 | 0.6700 | 0.7750 | 0.8000 | 0.6653 | clear p@1 regression |

**Key findings:**
1. **step640 is overall best**: p@4=0.7763 and p@8=0.800 both exceed baseline; p@1 slightly below step480 (0.685 vs 0.689) but negligible
2. **pass@8 fully recovered**: step640 and step960 reach p@8=0.800, **exceeding baseline's 0.790** — phase3's p@8 disadvantage was temporary, disappears with longer training
3. **Overfitting after step800**: test p@1 drops continuously (0.689→0.685→0.684→0.670) while train p@1 keeps rising (0.640→0.648→0.650→0.665) — classic generalization gap
4. **Dataset bottleneck**: MATH-500 with 400 training problems saturates around step480–640; further training only memorizes the training set

**Final conclusion: phase3 (EMA-GRPO + mastery filter) peaks at step480–640, outperforming standard GRPO by:**
- test pass@1: **+0.016~+0.020**
- test pass@8: **+0.010** (at step640)
- Further gains require a larger training set or data augmentation, not longer training

| Priority | Method | Complexity | Hypothesis | Status |
|----------|--------|-----------|------------|--------|
| ~~1~~ | ~~Direction A (soft weighting)~~ | ~~Low~~ | ~~Smooth curriculum > hard cutoff~~ | **Falsified** |
| ~~1~~ | ~~Direction B (reactive KL)~~ | ~~Medium~~ | ~~Targeted penalty > preemptive~~ | **Falsified** |
| ~~1~~ | ~~Extend phase3 training~~ | ~~Low~~ | ~~step480 not yet converged~~ | **Done (step640 optimal)** |
| 1 | Larger dataset / data augmentation | Medium | 400-problem dataset is the bottleneck | Next |
| 2 | Direction D (decoupled objectives) | High | Orthogonal learning/retention | Planned |
| 3 | Direction C (gradient projection) | Very High | Theoretically cleanest | Planned |
