"""Verification metrics for face embeddings."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


def compute_verification_metrics(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    labels: torch.Tensor,
    *,
    thresholds: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Evaluate verification performance using cosine similarity.

    Args:
        embeddings_a: Tensor of shape (N, D)
        embeddings_b: Tensor of shape (N, D)
        labels: Binary tensor where 1 indicates same identity.
        thresholds: Optional iterable of thresholds to evaluate. Defaults to [-1, 1] range.

    Returns:
        Dictionary containing best accuracy, optimal threshold, and distance statistics.
    """

    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError("Embeddings must have matching shape.")
    if embeddings_a.shape[0] != labels.shape[0]:
        raise ValueError("Labels must be same length as embeddings.")

    sims = F.cosine_similarity(embeddings_a, embeddings_b)

    if thresholds is None:
        thresholds = torch.linspace(-1.0, 1.0, steps=200, device=sims.device)
    else:
        thresholds = torch.tensor(
            list(thresholds), device=sims.device, dtype=sims.dtype
        )

    labels_float = labels.float()
    best_acc = torch.tensor(0.0, device=sims.device)
    best_thr = thresholds[0]

    for thr in thresholds:
        preds = (sims >= thr).float()
        acc = (preds == labels_float).float().mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    with torch.no_grad():
        preds = (sims >= best_thr).float()
        true_accepts = ((preds == 1) & (labels_float == 1)).float().sum()
        false_accepts = ((preds == 1) & (labels_float == 0)).float().sum()
        true_rejects = ((preds == 0) & (labels_float == 0)).float().sum()
        false_rejects = ((preds == 0) & (labels_float == 1)).float().sum()

        tar = true_accepts / (true_accepts + false_rejects + 1e-8)
        far = false_accepts / (false_accepts + true_rejects + 1e-8)

    return {
        "best_accuracy": best_acc.item(),
        "best_threshold": best_thr.item(),
        "tar": tar.item(),
        "far": far.item(),
        "mean_similarity": sims.mean().item(),
        "std_similarity": sims.std(unbiased=False).item(),
    }
