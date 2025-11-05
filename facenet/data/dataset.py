"""Dataset utilities for generating triplets from a face identity dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from PIL import Image
from torch.utils.data import Dataset


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_paths(root: Path, *, min_images_per_class: int) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = [
            path
            for path in sorted(class_dir.glob("*"))
            if path.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if len(images) >= min_images_per_class:
            class_to_images[class_dir.name] = images
    if not class_to_images:
        raise ValueError(
            f"No classes with at least {min_images_per_class} images were found in {root}"
        )
    return class_to_images


@dataclass
class TripletSample:
    anchor: Path
    positive: Path
    negative: Path
    anchor_label: str
    negative_label: str


class TripletFaceDataset(Dataset):
    """Dataset that produces anchor, positive, negative samples for triplet loss training."""

    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable[[Image.Image], object]] = None,
        min_images_per_class: int = 2,
        classes: Optional[Iterable[str]] = None,
        num_triplets: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.transform = transform
        full_mapping = find_image_paths(
            self.root, min_images_per_class=min_images_per_class
        )
        if classes is not None:
            missing = set(classes) - set(full_mapping.keys())
            if missing:
                raise ValueError(
                    f"Requested classes not present in dataset: {sorted(missing)}"
                )
            self.class_to_images = {name: full_mapping[name] for name in classes}
        else:
            self.class_to_images = full_mapping
        self.class_names = sorted(self.class_to_images.keys())
        self.num_triplets = num_triplets or sum(
            len(paths) for paths in self.class_to_images.values()
        )

        self._ensure_positive_candidates()

    def _ensure_positive_candidates(self) -> None:
        for class_name, images in self.class_to_images.items():
            if len(images) < 2:
                raise ValueError(
                    f"Class {class_name} contains fewer than two images. "
                    "Triplet generation requires at least anchor/positive pairs."
                )

    def __len__(self) -> int:
        return self.num_triplets

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        sample = self._sample_triplet()
        anchor = self._load_image(sample.anchor)
        positive = self._load_image(sample.positive)
        negative = self._load_image(sample.negative)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "anchor_label": sample.anchor_label,
            "negative_label": sample.negative_label,
        }

    def _sample_triplet(self) -> TripletSample:
        anchor_label = random.choice(self.class_names)
        positive_paths = self.class_to_images[anchor_label]
        anchor_path, positive_path = random.sample(positive_paths, 2)

        negative_label = random.choice(
            [name for name in self.class_names if name != anchor_label]
        )
        negative_path = random.choice(self.class_to_images[negative_label])

        return TripletSample(
            anchor=anchor_path,
            positive=positive_path,
            negative=negative_path,
            anchor_label=anchor_label,
            negative_label=negative_label,
        )

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")
