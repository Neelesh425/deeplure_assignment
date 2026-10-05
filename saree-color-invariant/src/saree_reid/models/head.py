"""
Classification head for ArcFace/CosFace metric learning.

Why ArcFace: adds angular margin to softmax, pushing embeddings of the same
design closer and different designs further apart in cosine space. This is
more effective than raw cross-entropy for learning discriminative embeddings
for retrieval tasks (Deng et al., 2019).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head (additive angular margin).

    Operates on L2-normalised embeddings. During training, adds angular margin
    to the target logit before softmax → pushes decision boundary.
    At inference, only the embedder is used (no head needed).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.3,
    ):
        super().__init__()
        self.s = s  # scale factor
        self.m = m  # angular margin
        self.num_classes = num_classes

        # Weight matrix: each column is the class "proxy" on the unit hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Threshold to prevent numerical instability
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ArcFace logits.

        Args:
            embeddings: (B, D) L2-normalised embeddings
            labels: (B,) integer class labels
        Returns:
            (B, num_classes) scaled logits with angular margin
        """
        # Normalise weights
        w = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity = dot product of normalised vectors
        cosine = F.linear(embeddings, w)  # (B, num_classes)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # ArcFace: cos(theta + m) for target class
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # Numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to target class
        one_hot = F.one_hot(labels, self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return logits * self.s  # scale


def build_head(cfg: dict[str, Any], num_classes: int) -> ArcFaceHead:
    """Build ArcFace head from config."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    return ArcFaceHead(
        embedding_dim=model_cfg.get("embedding_dim", 128),
        num_classes=num_classes,
        s=train_cfg.get("arcface_s", 30.0),
        m=train_cfg.get("arcface_m", 0.3),
    )
