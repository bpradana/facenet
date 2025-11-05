"""PyTorch Lightning DataModule for face recognition training."""

from __future__ import annotations

import random
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from .dataset import TripletFaceDataset
from .identity_dataset import FaceIdentityDataset
from .sampler import BalancedBatchSampler
from .transforms import build_eval_transform, build_train_transform
from ..config import DataConfig


class FaceDataModule(L.LightningDataModule):
    """LightningDataModule that handles train/validation splits by identity."""

    def __init__(self, config: DataConfig, *, seed: int = 42) -> None:
        super().__init__()
        self.cfg = config
        self.seed = seed

        expected_batch = config.classes_per_batch * config.samples_per_class
        if expected_batch != config.batch_size:
            raise ValueError(
                "Batch size must equal classes_per_batch * samples_per_class "
                f"({config.classes_per_batch} * {config.samples_per_class} = {expected_batch}). "
                f"Got batch_size={config.batch_size}."
            )
        if config.samples_per_class < 2:
            raise ValueError(
                "samples_per_class must be at least 2 for batch-hard mining."
            )

        self._train_dataset: Optional[FaceIdentityDataset] = None
        self._train_sampler: Optional[BalancedBatchSampler] = None
        self._val_dataset: Optional[TripletFaceDataset] = None
        self._identity_split: Optional[tuple[list[str], list[str]]] = None
        self._pin_memory = self._resolve_pin_memory()
        self._persistent_workers = (
            self.cfg.persistent_workers and self.cfg.num_workers > 0
        )

    def prepare_data(self) -> None:  # pragma: no cover - Lightning hook
        # Nothing to download, but validate directory exists.
        TripletFaceDataset(
            self.cfg.root,
            transform=build_eval_transform(self.cfg.image_size),
            min_images_per_class=self.cfg.min_images_per_class,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_dataset is not None and self._val_dataset is not None:
            return

        train_transform = build_train_transform(
            self.cfg.image_size, augmentations=self.cfg.augmentations
        )
        eval_transform = build_eval_transform(self.cfg.image_size)

        train_dataset = FaceIdentityDataset(
            self.cfg.root,
            transform=train_transform,
            min_images_per_class=self.cfg.min_images_per_class,
        )
        class_names = train_dataset.class_names.copy()

        rng = random.Random(self.seed)
        rng.shuffle(class_names)

        if len(class_names) < self.cfg.classes_per_batch:
            raise ValueError(
                "Not enough identities to satisfy classes_per_batch. "
                f"Required {self.cfg.classes_per_batch}, found {len(class_names)}."
            )

        split_index = max(1, int(len(class_names) * self.cfg.val_split))
        val_classes = class_names[:split_index]
        train_classes = class_names[split_index:] or class_names
        self._identity_split = (train_classes, val_classes)

        if len(train_classes) < self.cfg.classes_per_batch:
            raise ValueError(
                "Not enough training identities after split to satisfy classes_per_batch. "
                f"Required {self.cfg.classes_per_batch}, found {len(train_classes)}."
            )

        # Filter validation dataset to val classes (or all classes if split small)
        self._val_dataset = TripletFaceDataset(
            self.cfg.root,
            transform=eval_transform,
            min_images_per_class=self.cfg.min_images_per_class,
            classes=val_classes if val_classes else class_names,
        )

        if val_classes:
            allowed_class_set = set(train_classes)
            filtered_samples = [
                sample
                for sample in train_dataset.samples
                if sample.label_name in allowed_class_set
            ]
            if not filtered_samples:
                raise ValueError(
                    "No training samples remain after applying train/val split."
                )

            class_remap = {name: idx for idx, name in enumerate(sorted(train_classes))}
            new_samples = []
            new_mapping: dict[int, list[int]] = {
                idx: [] for idx in class_remap.values()
            }

            for sample in filtered_samples:
                new_label = class_remap[sample.label_name]
                new_samples.append(
                    type(sample)(
                        path=sample.path, label=new_label, label_name=sample.label_name
                    )
                )
                new_mapping[new_label].append(len(new_samples) - 1)

            train_dataset.samples = new_samples
            train_dataset.label_to_indices = new_mapping
            train_dataset.class_names = sorted(train_classes)

        self._train_dataset = train_dataset
        self._train_sampler = BalancedBatchSampler(
            train_dataset.label_to_indices,
            classes_per_batch=self.cfg.classes_per_batch,
            samples_per_class=self.cfg.samples_per_class,
            dataset_size=len(train_dataset),
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        assert self._train_sampler is not None
        return DataLoader(
            self._train_dataset,
            batch_sampler=self._train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=False,
        )

    @property
    def split_identities(self) -> Optional[tuple[list[str], list[str]]]:
        return self._identity_split

    def _resolve_pin_memory(self) -> bool:
        """Enable pin_memory only when supported."""

        if not self.cfg.pin_memory:
            return False
        if torch.cuda.is_available():
            return True
        # Disable for MPS/CPU where pinned memory brings no benefit.
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return False
        return False
