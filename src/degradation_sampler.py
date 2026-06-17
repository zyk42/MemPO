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
"""Degradation-aware prompt replay sampler for EMA-GRPO.

Oversamples prompts where the model struggles (low EMA accuracy) or has
regressed below its initial performance.  Two complementary signals:

  difficulty_score[p]   = max(0, 1 - ema_mean[p])
      Captures prompts the model has always found hard (never learned).

  degradation_score[p]  = max(0, initial_acc[p] - ema_mean[p])
      Captures prompts the model previously solved but has since forgotten.

Combined sampling weight:

    sample_weight[p] = 1 + gamma * (difficulty_score[p] + degradation_score[p])

Prompts with no history (memory bank warmup) keep weight=1.0 (uniform).
Automatically integrates with the AbstractCurriculumSampler interface:
  sampler.update(batch) is called after each training step by ray_trainer.

Usage note: set data.dataloader_num_workers=0 to avoid prefetch/weight mismatch.
"""

import numpy as np
import torch
from collections.abc import Sized

from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class DegradationAwareSampler(AbstractCurriculumSampler):
    """Weighted sampler that oversamples hard and regressed prompts.

    Combines two signals from PromptMemoryBank:
      - difficulty  = 1 - ema_mean  (model still struggling, possibly never learned)
      - degradation = max(0, initial_acc - ema_mean)  (previously solved, now forgotten)

    weight[p] = 1 + gamma * (difficulty_score[p] + degradation_score[p])

    Args:
        data_source: The training dataset (must be Sized).
        data_config: verl data config (required by interface; not used internally).
        memory_bank: PromptMemoryBank instance to query scores.
        gamma: Multiplier for the combined boost.
            gamma=0 → uniform sampling (no-op).
            gamma=1 → e.g. difficulty=0.8 + degradation=0.2 → weight=2.0 (2x sampling).
    """

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
        memory_bank=None,
        gamma: float = 1.0,
    ):
        self.data_source = data_source
        self.memory_bank = memory_bank
        self.gamma = gamma
        self._n = len(data_source)
        # All weights start at 1.0 (uniform) until memory bank accumulates history
        self._weights = torch.ones(self._n, dtype=torch.float32)

    def update(self, batch: DataProto) -> None:
        """Refresh sampling weights for all prompts in the current batch.

        Called by ray_trainer after each training step. Only updates the prompts
        present in the batch; unvisited prompts retain their current weight.
        """
        if self.memory_bank is None or self.gamma == 0.0:
            return

        ntb = batch.non_tensor_batch
        # Extract stable dataset indices (same logic as compute_advantage in ray_trainer)
        if "index" in ntb:
            indices = ntb["index"]
        elif "extra_info" in ntb:
            extra = ntb["extra_info"]
            indices = [ei["index"] if isinstance(ei, dict) else int(ei) for ei in extra]
        else:
            return

        # Update weight for each unique prompt seen in this batch
        for idx in set(int(i) for i in indices):
            if 0 <= idx < self._n:
                diff = self.memory_bank.difficulty_score(idx)
                deg = self.memory_bank.degradation_score(idx)
                self._weights[idx] = 1.0 + self.gamma * (diff + deg)

    def __iter__(self):
        # Weighted sampling with replacement over the full dataset
        indices = torch.multinomial(self._weights, self._n, replacement=True)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self._n

    def state_dict(self) -> dict:
        return {"weights": self._weights.tolist()}

    def load_state_dict(self, state: dict) -> None:
        weights = state.get("weights")
        if weights is not None and len(weights) == self._n:
            self._weights = torch.tensor(weights, dtype=torch.float32)
