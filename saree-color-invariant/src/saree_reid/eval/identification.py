"""
Identification metrics: CMC (Rank-1/5/10) and mAP on cosine similarity.

Protocol: embed all gallery and query images, compute pairwise cosine
similarity, rank gallery by similarity to each query, compute CMC and mAP.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def compute_cosine_similarity_matrix(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
) -> np.ndarray:
    """Cosine similarity matrix between query and gallery embeddings."""
    # Already L2-normalised → dot product = cosine similarity
    return query_emb @ gallery_emb.T


def compute_cmc(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_vals: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Compute CMC curve (Cumulative Matching Characteristic).

    For each query, check if the correct design appears in the top-k
    gallery retrievals.
    """
    n_queries = similarity_matrix.shape[0]
    results = {}

    # Rank gallery for each query (descending similarity)
    ranked_indices = np.argsort(-similarity_matrix, axis=1)

    for k in k_vals:
        hits = 0
        for q_idx in range(n_queries):
            top_k_gallery = ranked_indices[q_idx, :k]
            top_k_labels = gallery_labels[top_k_gallery]
            if query_labels[q_idx] in top_k_labels:
                hits += 1
        results[f"rank_{k}"] = hits / n_queries

    return results


def compute_map(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """
    Compute mean Average Precision.

    For each query, compute AP = area under precision-recall curve
    for retrieving all same-design gallery images.
    """
    n_queries = similarity_matrix.shape[0]
    aps = []

    ranked_indices = np.argsort(-similarity_matrix, axis=1)

    for q_idx in range(n_queries):
        q_label = query_labels[q_idx]
        ranked_labels = gallery_labels[ranked_indices[q_idx]]

        # Relevant = same design
        relevant = (ranked_labels == q_label)
        n_relevant = relevant.sum()

        if n_relevant == 0:
            continue  # skip queries with no gallery match

        # Compute AP
        cum_relevant = np.cumsum(relevant)
        precision_at_k = cum_relevant / np.arange(1, len(relevant) + 1)
        ap = (precision_at_k * relevant).sum() / n_relevant
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract embeddings, labels, and paths from a dataloader."""
    model.eval()
    all_embs = []
    all_labels = []
    all_paths = []

    for images, labels, metas in dataloader:
        images = images.to(device)
        embs = model(images)
        all_embs.append(embs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_paths.extend(metas["path"])

    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.array(all_labels)
    return embeddings, labels, all_paths


def run_identification(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    k_vals: list[int] = [1, 5, 10],
) -> dict[str, Any]:
    """
    Full identification evaluation.

    Since each image is its own design, we use leave-one-out:
    each image is a query, the rest are the gallery.
    """
    embeddings, labels, paths = extract_embeddings(model, dataloader, device)

    # Leave-one-out: each image is query, rest are gallery
    n = len(embeddings)
    if n < 2:
        return {"error": "not_enough_images"}

    # Full similarity matrix
    sim_matrix = compute_cosine_similarity_matrix(embeddings, embeddings)

    # Zero out self-similarity (query shouldn't match itself)
    np.fill_diagonal(sim_matrix, -1.0)

    # CMC and mAP
    cmc = compute_cmc(sim_matrix, labels, labels, k_vals)
    mAP = compute_map(sim_matrix, labels, labels)

    results = {
        "num_images": n,
        "num_designs": len(np.unique(labels)),
        "mAP": round(mAP, 4),
        **{k: round(v, 4) for k, v in cmc.items()},
    }

    log.info(f"Identification: mAP={mAP:.4f}, " +
             ", ".join(f"{k}={v:.4f}" for k, v in cmc.items()))
    return results
