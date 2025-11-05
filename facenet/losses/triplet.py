"""Triplet loss implementation."""

from __future__ import annotations

from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Classic margin-based triplet loss."""

    def __init__(self, margin: float = 0.5, *, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("Reduction must be either 'mean' or 'sum'")
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):  # type: ignore[override]
        pos_distance = F.pairwise_distance(anchor, positive, keepdim=True)
        neg_distance = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = F.relu(pos_distance - neg_distance + self.margin)
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()
