"""Identity-based dataset for batch-hard triplet mining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import find_image_paths


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


@dataclass
class DatasetSample:
    path: Path
    label: int
    label_name: str


class FaceIdentityDataset(Dataset):
    """Dataset returning individual images and labels."""

    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        min_images_per_class: int = 2,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.transform = transform
        self.class_to_images = find_image_paths(
            self.root, min_images_per_class=min_images_per_class
        )
        self.class_names = sorted(self.class_to_images.keys())

        self.samples: List[DatasetSample] = []
        self.label_to_indices: Dict[int, List[int]] = {}
        for label, class_name in enumerate(self.class_names):
            images = self.class_to_images[class_name]
            start_index = len(self.samples)
            for offset, path in enumerate(images):
                self.samples.append(
                    DatasetSample(path=path, label=label, label_name=class_name)
                )
            indices = list(range(start_index, len(self.samples)))
            self.label_to_indices[label] = indices

        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:  # type: ignore[override]
        sample = self.samples[index]
        image = _load_image(sample.path)
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "label_name": sample.label_name,
        }
