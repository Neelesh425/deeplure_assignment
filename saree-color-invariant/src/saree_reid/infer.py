"""
Inference: identify (query → top-k from gallery) and verify (pair → decision).

Post-processing: L2 normalisation is baked into the model's forward pass,
cosine similarity = dot product of normalised embeddings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from saree_reid.data.transforms import build_eval_transform
from saree_reid.eval.identification import compute_cosine_similarity_matrix
from saree_reid.models.backbone import build_model
from saree_reid.utils import get_device, load_config

log = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    cfg: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cfg is None:
        cfg = ckpt.get("cfg", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def embed_image(
    model: torch.nn.Module,
    image_path: str | Path,
    transform: Any,
    device: torch.device,
) -> np.ndarray:
    """Embed a single image → (embedding_dim,) numpy array."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    emb = model(tensor).cpu().numpy()[0]
    return emb


def identify(
    model: torch.nn.Module,
    query_path: str | Path,
    gallery_paths: list[str | Path],
    transform: Any,
    device: torch.device,
    top_k: int = 5,
) -> list[dict]:
    """
    Identify: rank gallery by similarity to query.

    Returns list of top-k matches with paths and similarity scores.
    """
    query_emb = embed_image(model, query_path, transform, device)

    gallery_embs = []
    for gp in gallery_paths:
        gallery_embs.append(embed_image(model, gp, transform, device))
    gallery_embs = np.stack(gallery_embs)

    # Cosine similarity (already L2-normalised)
    sims = query_emb @ gallery_embs.T

    top_indices = np.argsort(-sims)[:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "path": str(gallery_paths[idx]),
            "similarity": round(float(sims[idx]), 4),
        })

    return results


def verify(
    model: torch.nn.Module,
    path_a: str | Path,
    path_b: str | Path,
    transform: Any,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Verify: decide if two images show the same design.

    Returns similarity score and binary decision.
    """
    emb_a = embed_image(model, path_a, transform, device)
    emb_b = embed_image(model, path_b, transform, device)

    similarity = float(emb_a @ emb_b)
    decision = similarity >= threshold

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "same_design": bool(decision),
    }
