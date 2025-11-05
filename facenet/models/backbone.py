"""Backbone network factory for FaceNet embeddings."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models

BackboneName = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
]


_BACKBONE_FACTORY = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
}


_BACKBONE_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
    "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
    "resnext50_32x4d": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    "resnext101_32x8d": models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
}


class FaceEmbeddingNet(nn.Module):
    """ResNet backbone with projection head for embedding generation."""

    def __init__(
        self,
        backbone_name: BackboneName,
        embedding_dim: int,
        *,
        pretrained: bool = True,
        dropout: float = 0.0,
        train_backbone: bool = True,
    ) -> None:
        super().__init__()

        if backbone_name not in _BACKBONE_FACTORY:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        weights = _BACKBONE_WEIGHTS[backbone_name] if pretrained else None
        backbone = _BACKBONE_FACTORY[backbone_name](weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        projection_layers: list[nn.Module] = []
        if dropout > 0.0:
            projection_layers.append(nn.Dropout(p=dropout))
        projection_layers.append(nn.Linear(in_features, embedding_dim))

        self.backbone = backbone
        self.projection = nn.Sequential(*projection_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(x)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


def build_backbone(
    backbone_name: BackboneName,
    embedding_dim: int,
    *,
    pretrained: bool,
    dropout: float,
    train_backbone: bool,
) -> FaceEmbeddingNet:
    """Factory function used by the Lightning module."""

    return FaceEmbeddingNet(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        dropout=dropout,
        train_backbone=train_backbone,
    )
