"""
Verification metrics: ROC-AUC, EER, TAR@FAR.

Given image pairs, compute similarity and evaluate binary classification
(same design or not).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

log = logging.getLogger(__name__)


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate and its threshold."""
    fnr = 1 - tpr
    # Find point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def compute_tar_at_far(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_far: float,
) -> float:
    """Compute TAR (True Accept Rate) at a given FAR."""
    # Find the threshold where FAR ≈ target_far
    idx = np.searchsorted(fpr, target_far)
    if idx >= len(tpr):
        idx = len(tpr) - 1
    return float(tpr[idx])


def run_verification(
    similarities: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Full verification evaluation.

    Args:
        similarities: (N,) cosine similarities for each pair
        labels: (N,) binary labels (1=same design, 0=different)
        threshold: if None, compute optimal from EER
    """
    if len(similarities) < 2:
        return {"error": "not_enough_pairs"}

    # ROC
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc = roc_auc_score(labels, similarities)

    # EER
    eer, eer_thresh = compute_eer(fpr, tpr, thresholds)

    # TAR@FAR
    tar_at_1e2 = compute_tar_at_far(fpr, tpr, 1e-2)
    tar_at_1e3 = compute_tar_at_far(fpr, tpr, 1e-3)

    # Accuracy at threshold
    if threshold is None:
        threshold = eer_thresh  # use EER threshold

    predictions = (similarities >= threshold).astype(int)
    accuracy = float((predictions == labels).mean())

    results = {
        "auc": round(float(auc), 4),
        "eer": round(eer, 4),
        "eer_threshold": round(float(eer_thresh), 4),
        "tar_at_far_1e2": round(tar_at_1e2, 4),
        "tar_at_far_1e3": round(tar_at_1e3, 4),
        "accuracy_at_threshold": round(accuracy, 4),
        "threshold_used": round(float(threshold), 4),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
    }

    log.info(f"Verification: AUC={auc:.4f}, EER={eer:.4f}, "
             f"TAR@FAR=1e-2: {tar_at_1e2:.4f}, accuracy={accuracy:.4f}")
    return results
