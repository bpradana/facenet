"""Samplers for balanced identity batches."""

from __future__ import annotations

import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):
    """Samples batches with a fixed number of identities and samples per identity."""

    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        *,
        classes_per_batch: int,
        samples_per_class: int,
        dataset_size: int,
    ) -> None:
        if classes_per_batch <= 0 or samples_per_class <= 0:
            raise ValueError("classes_per_batch and samples_per_class must be positive")

        self.label_to_indices = {k: v[:] for k, v in label_to_indices.items()}
        self.classes = list(self.label_to_indices.keys())
        if len(self.classes) < classes_per_batch:
            raise ValueError(
                f"Requested {classes_per_batch} classes per batch, "
                f"but only {len(self.classes)} classes available."
            )

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = classes_per_batch * samples_per_class
        self.dataset_size = dataset_size
        self.batches_per_epoch = max(1, dataset_size // self.batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.batches_per_epoch):
            batch: List[int] = []
            selected_classes = random.sample(self.classes, self.classes_per_batch)
            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                if not indices:
                    continue
                if len(indices) >= self.samples_per_class:
                    batch.extend(random.sample(indices, self.samples_per_class))
                else:
                    reps = self.samples_per_class - len(indices)
                    batch.extend(indices)
                    batch.extend(random.choices(indices, k=reps))
            if len(batch) != self.batch_size:
                # If some classes were missing, pad from other classes.
                deficit = self.batch_size - len(batch)
                fallback_indices = random.choices(range(self.dataset_size), k=deficit)
                batch.extend(fallback_indices)
            yield batch
