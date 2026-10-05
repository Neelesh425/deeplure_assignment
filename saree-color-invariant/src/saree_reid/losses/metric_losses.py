"""
Metric learning losses: ArcFace cross-entropy wrapper.

Why ArcFace over triplet loss: ArcFace uses all classes in each batch
(via proxy weights), not just triplet pairs. More stable training, faster
convergence, and better generalisation on small datasets where mining hard
triplets is noisy. The angular margin directly optimises cosine similarity,
which is our retrieval metric.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    """Cross-entropy on ArcFace logits (the margin is in the head)."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)
