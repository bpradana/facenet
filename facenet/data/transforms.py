"""Image preprocessing pipelines."""

from __future__ import annotations


from torchvision import transforms


def build_train_transform(
    image_size: int, *, augmentations: bool = True
) -> transforms.Compose:
    augmentation_steps = []
    if augmentations:
        augmentation_steps = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
                p=0.3,
            ),
            transforms.RandomRotation(degrees=10),
        ]

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            *augmentation_steps,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
