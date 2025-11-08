"""Backbone network factory for FaceNet embeddings."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torchvision.models import get_model, get_model_weights

BackboneName = str


class FaceEmbeddingNet(nn.Module):
    """Backbone with projection head for embedding generation."""

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

        backbone = _load_backbone(backbone_name, pretrained)
        in_features = _strip_classifier(backbone)

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


def _load_backbone(name: str, pretrained: bool) -> nn.Module:
    try:
        weights_enum = get_model_weights(name)
        weights = weights_enum.DEFAULT if pretrained else None
    except (ValueError, RuntimeError):
        weights = None if not pretrained else None
    try:
        return get_model(name, weights=weights)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Unsupported backbone '{name}': {exc}") from exc


def _strip_classifier(backbone: nn.Module) -> int:
    parent, key, classifier_module = _locate_classifier(backbone)
    in_features = _extract_in_features(classifier_module)

    if isinstance(key, int):
        parent[key] = nn.Identity()
    else:
        setattr(parent, key, nn.Identity())

    return in_features


def _locate_classifier(backbone: nn.Module) -> Tuple[nn.Module, int | str, nn.Module]:
    candidate_attrs: Iterable[str] = (
        "fc",
        "classifier",
        "head",
        "heads",
        "last_linear",
        "logits",
        "linear",
    )
    for attr in candidate_attrs:
        if hasattr(backbone, attr):
            module = getattr(backbone, attr)
            if isinstance(module, nn.Sequential):
                idx = _find_last_linear_index(module)
                if idx is not None:
                    return module, idx, module[idx]
            elif isinstance(module, nn.Linear):
                return backbone, attr, module

    name, module = _find_last_linear(backbone)
    parent, key = _resolve_parent(backbone, name)
    return parent, key, module


def _find_last_linear_index(seq: nn.Sequential) -> int | None:
    for idx in range(len(seq) - 1, -1, -1):
        if isinstance(seq[idx], nn.Linear):
            return idx
    return None


def _find_last_linear(backbone: nn.Module) -> Tuple[str, nn.Linear]:
    last: Tuple[str, nn.Linear] | None = None
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear):
            last = (name, module)
    if last is None:
        raise ValueError("Backbone does not contain any linear layers to strip.")
    return last


def _resolve_parent(backbone: nn.Module, name: str) -> Tuple[nn.Module, int | str]:
    parts = name.split(".")
    parent: nn.Module = backbone
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    key_part = parts[-1]
    key: int | str
    if key_part.isdigit():
        key = int(key_part)
    else:
        key = key_part
    return parent, key


def _extract_in_features(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return module.in_features
    if isinstance(module, nn.Sequential):
        for layer in reversed(list(module.children())):
            if isinstance(layer, nn.Linear):
                return layer.in_features
    raise ValueError("Classifier module does not expose a linear layer.")
