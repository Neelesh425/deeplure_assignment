"""
Baselines to prove the evaluation is meaningful.

1. Colour histogram baseline: should FAIL the colour-invariance requirement
   (proves our eval catches colour-dependent models).
2. ImageNet zero-shot: pretrained features without fine-tuning.
3. Grayscale zero-shot: pretrained features on grayscale input.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


def colour_histogram_embedding(
    images: list[np.ndarray],
    n_bins: int = 32,
) -> np.ndarray:
    """
    Compute colour histogram 'embedding' for each image.
    This baseline should FAIL: it matches by colour, not design.
    """
    embeddings = []
    for img in images:
        # img: (H, W, 3) uint8
        hist_r = np.histogram(img[:, :, 0], bins=n_bins, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1], bins=n_bins, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2], bins=n_bins, range=(0, 256))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        # L2 normalise
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        embeddings.append(hist)
    return np.stack(embeddings)


@torch.no_grad()
def imagenet_zeroshot_embedding(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from untrained (ImageNet-pretrained) model."""
    model.eval()
    all_embs = []
    all_labels = []
    for images, labels, _ in dataloader:
        embs = model(images.to(device))
        all_embs.append(embs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_embs), np.array(all_labels)
