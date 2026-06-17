# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-prompt EMA statistics bank for cross-epoch advantage normalization (EMA-GRPO).

Maintains an exponential moving average of per-prompt reward mean and std across
training steps, enabling stable advantage normalization that is not subject to the
high variance of single-batch estimates.
"""

import json
import os
from typing import Optional


class PromptMemoryBank:
    """Maintains per-prompt EMA statistics across training steps.

    For each prompt (identified by its dataset index), stores:
      - EMA mean of observed rewards
      - Number of times this prompt has been seen (for warm-up logic)

    Uses parameter-free dynamic alpha derived from rollout_n:
      α_t(p) = 1/n + (n-2)/n · (1 - |μ̂_t(p) - μ_{t-1}(p)|)

    Where:
      - α_min = 1/n  (trust current batch fully when large deviation)
      - α_max = (n-1)/n  (trust history fully when no deviation)
      - δ = |μ̂_t - μ_{t-1}| ∈ [0, 1]

    Mastery threshold is also derived from n: τ = (n-1)/n.
    Filtering triggers when batch_mean > τ (strictly greater than, i.e. all n correct).

    Args:
        rollout_n (int): Number of rollouts per prompt (e.g. 8). Used to derive
            both the dynamic alpha bounds [1/n, (n-1)/n] and the mastery threshold τ=(n-1)/n.
        warmup_steps (int): Number of appearances before switching from
            batch statistics to EMA statistics. Minimum safe value is 1:
            on the very first encounter (count=0) query() returns 0.0 for
            unseen prompts, so warmup_steps=0 would use 0 as baseline and
            inflate advantages. From count=1 onward the EMA is already
            initialized from the first observed batch mean, so warmup_steps=1
            is sufficient.
        eps (float): Small constant to avoid division by zero.
    """

    def __init__(
        self,
        rollout_n: int = 8,
        warmup_steps: int = 1,
        eps: float = 1e-6,
        kl_gamma: float = 0.0,
        mastery_soft_alpha: float = 0.0,
    ):
        self.rollout_n = rollout_n
        self.warmup_steps = warmup_steps
        self.eps = eps
        # mastery_threshold: derived from rollout_n as (n-1)/n.
        # Filtering triggers when batch_mean > threshold (strictly greater),
        # meaning all n rollouts are correct.
        self.mastery_threshold = (rollout_n - 1) / rollout_n
        # kl_gamma: KL multiplier scale. When > 0, prompts with high EMA mean get a stronger
        # KL penalty: beta_eff = beta * (1 + kl_gamma * ema_mean). Set to 0 to disable.
        self.kl_gamma = kl_gamma
        # mastery_soft_alpha: when > 0, replaces the hard mastery filter with a soft advantage
        # weighting: advantage *= (1 - ema_mean) ** mastery_soft_alpha.
        # This creates a smooth learning curriculum: mastered prompts get smaller gradients
        # but are not completely zeroed out, preventing capability loss (pass@k regression).
        # Set to 0.0 to use the original hard mastery filter instead.
        self.mastery_soft_alpha = mastery_soft_alpha
        # key: dataset_index (int or str) -> (ema_mean, count)
        self._bank: dict[str, tuple[float, int]] = {}
        # key: dataset_index -> initial batch_mean recorded on first encounter.
        # Used for reactive KL: only boost KL when current performance drops
        # below the initial model's performance on that prompt.
        self._initial_acc: dict[str, float] = {}

    def _key(self, dataset_index) -> str:
        return str(int(dataset_index))

    def query(self, dataset_index) -> tuple[float, float, int]:
        """Return (ema_mean, dummy_std, count) for the given dataset index.

        std is no longer tracked (always returns 1.0). Advantage normalization
        is purely mean-subtraction: A = r - ema_mean, no division by std.
        Returns (0.0, 1.0, 0) for unseen prompts.
        """
        entry = self._bank.get(self._key(dataset_index))
        if entry is None:
            return (0.0, 1.0, 0)
        ema_mean, count = entry
        return (ema_mean, 1.0, count)

    def is_warmed_up(self, dataset_index) -> bool:
        """Return True if this prompt has been seen >= warmup_steps times."""
        _, _, count = self.query(dataset_index)
        return count >= self.warmup_steps

    def is_mastered_batch(self, mean: float) -> bool:
        """Return True if the given mean strictly exceeds the mastery threshold τ = (n-1)/n.

        Should be called with the updated EMA mean μ_t (after update()), so that mastery
        is confirmed only when the model has *consistently* solved this prompt over multiple
        steps — not just one lucky batch. When True, the caller falls back to batch mean as
        baseline → advantage = 0, restoring GRPO's zero-gradient protection.
        """
        return mean > self.mastery_threshold

    def soft_mastery_weight(self, dataset_index) -> float:
        """Return the soft advantage weight for this prompt: (1 - ema_mean) ** mastery_soft_alpha.

        Used when mastery_soft_alpha > 0 to scale down advantages for already-mastered prompts
        instead of hard-zeroing them (the original mastery filter behavior).

        Returns 1.0 when mastery_soft_alpha == 0 (no weighting).
        """
        if self.mastery_soft_alpha == 0.0:
            return 1.0
        ema_mean, _, _ = self.query(dataset_index)
        return max(0.0, 1.0 - ema_mean) ** self.mastery_soft_alpha

    def degradation_score(self, dataset_index) -> float:
        """Return how much the model has regressed on this prompt since first encounter.

        score = max(0, initial_acc[p] - ema_mean[p])

        Returns 0.0 during the single-step warmup (count < 1): on the very first
        encounter query() returns ema_mean=0.0 (uninitialized), which would make
        every unseen prompt look maximally degraded. From count=1 onward the EMA
        is initialized from the first observed batch mean, so the estimate is valid.
        Returns 0.0 for unseen prompts.
        Used as the per-prompt sampling weight boost in DegradationAwareSampler.
        """
        key = self._key(dataset_index)
        if key not in self._initial_acc:
            return 0.0
        ema_mean, _, count = self.query(dataset_index)
        # Wait for EMA to converge before trusting regression signal
        if count < self.warmup_steps:
            return 0.0
        return max(0.0, self._initial_acc[key] - ema_mean)

    def difficulty_score(self, dataset_index) -> float:
        """Return the current difficulty of this prompt: 1 - ema_mean.

        High score = model still struggles on this prompt (low historical accuracy).
        Low score  = model has mastered this prompt (high historical accuracy).

        Returns 0.0 during the single-step warmup (count < 1): before the first
        update, ema_mean=0.0 would make every prompt look maximally hard and
        cause indiscriminate oversampling. From count=1 onward the EMA is
        initialized from the first observed batch mean and the estimate is valid.
        Returns 0.0 for unseen prompts.
        Used together with degradation_score in DegradationAwareSampler.
        """
        ema_mean, _, count = self.query(dataset_index)
        if count < self.warmup_steps:
            return 0.0
        return max(0.0, 1.0 - ema_mean)

    def kl_multiplier(self, dataset_index) -> float:
        """Return the per-prompt KL penalty multiplier.

        Reactive KL logic: only boost the KL penalty when the current EMA
        accuracy has dropped *below* the initial model's accuracy on this prompt.
        The boost is proportional to the regression depth:

            multiplier = 1 + kl_gamma * max(0, initial_acc - ema_mean)

        This ensures the KL reference policy (initial model) is always at least
        as good as the current policy on the penalized prompt, making the
        penalty meaningful and directionally correct.

        When ema_mean >= initial_acc (model is better than or equal to its
        starting point), the multiplier is 1.0 — no constraint applied.

        Returns 1.0 when kl_gamma == 0 (no boost).
        """
        if self.kl_gamma == 0.0:
            return 1.0
        ema_mean, _, _ = self.query(dataset_index)
        initial_acc = self._initial_acc.get(self._key(dataset_index), 0.0)
        regression = max(0.0, initial_acc - ema_mean)
        return 1.0 + self.kl_gamma * regression

    def update(self, dataset_index, batch_mean: float, batch_std: float):
        """Update EMA statistics with new batch observations.

        The variance update includes a bias-correction term that captures the
        deviation between the current batch mean and the updated EMA mean,
        preventing underestimation of true variance.

        Should be called AFTER computing advantages to avoid using future information.

        Args:
            dataset_index: Stable identifier for this prompt (dataset row index).
            batch_mean: Mean reward across rollouts in the current batch.
            batch_std: Std of rewards across rollouts in the current batch.
        """
        key = self._key(dataset_index)
        entry = self._bank.get(key)
        mu_old, count = entry if entry is not None else (0.0, 0)

        # Record initial model accuracy on the very first encounter.
        # This serves as the baseline for reactive KL: we only penalise
        # regression below this level (where the initial model is not worse).
        if count == 0 and key not in self._initial_acc:
            self._initial_acc[key] = batch_mean

        if count == 0:
            # Cold-start: initialize EMA mean directly from the first batch
            # observation (mean of the n rollouts for this prompt).
            mu_new = batch_mean
        else:
            # Dynamic alpha: parameter-free, derived solely from rollout_n.
            # α_t(p) = 1/n + (n-2)/n · (1 - |μ̂_t - μ_{t-1}|)
            #   - Large deviation (δ→1): α → 1/n  (trust current batch; discard stale history)
            #   - No deviation  (δ→0): α → (n-1)/n  (trust history; minor update)
            n = self.rollout_n
            delta = abs(batch_mean - mu_old)
            alpha = 1.0 / n + ((n - 2) / n) * (1.0 - delta)
            # Clamp to [1/n, (n-1)/n] to guard against delta > 1 (shouldn't happen, but safe)
            alpha = max(1.0 / n, min((n - 1) / n, alpha))
            mu_new = alpha * mu_old + (1.0 - alpha) * batch_mean

        # std is no longer tracked. Advantage = r - ema_mean (no division by std),
        # matching Dr. GRPO's finding that std normalization adds noise, while the
        # EMA mean provides a stable cross-step baseline.
        self._bank[key] = (mu_new, count + 1)

    def save(self, path: str):
        """Save the memory bank to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w") as f:
            json.dump({"bank": self._bank, "initial_acc": self._initial_acc}, f)

    def load(self, path: str):
        """Load the memory bank from a JSON file."""
        with open(path) as f:
            raw = json.load(f)
        # Support both old format (plain dict) and new format (with initial_acc)
        if "bank" in raw:
            raw_bank = raw["bank"]
            self._initial_acc = raw.get("initial_acc", {})
        else:
            # Legacy format: plain dict
            raw_bank = raw
            self._initial_acc = {}
        # Support old 3-tuple (mean, std, count) and new 2-tuple (mean, count)
        bank = {}
        for k, v in raw_bank.items():
            v = tuple(v)
            bank[k] = (v[0], v[-1])  # (mean, count): works for both (m,s,c) and (m,c)
        self._bank = bank

    def __len__(self) -> int:
        return len(self._bank)

    def __repr__(self) -> str:
        return (
            f"PromptMemoryBank(rollout_n={self.rollout_n}, warmup_steps={self.warmup_steps}, "
            f"mastery_threshold={self.mastery_threshold:.4f}, kl_gamma={self.kl_gamma}, "
            f"mastery_soft_alpha={self.mastery_soft_alpha}, size={len(self._bank)})"
        )
