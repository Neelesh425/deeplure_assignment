"""
Embedding backbone: lightweight pretrained CNN + GeM pooling + projection head.

Architecture choice: MobileNetV3-Small
- Why: smallest viable backbone (~2.5M params), fast inference, proven on
  texture/fine-grained tasks. Good accuracy/efficiency Pareto point.
- Alternatives considered: EfficientNet-B0 (5.3M, heavier), ResNet-18 (11.7M),
  ConvNeXt-Tiny (28.6M, too large), DINOv2-S (21M, ViT overhead on small data).
- GeM pooling: learnable generalised mean pooling captures texture patterns
  better than global average pooling (standard in image retrieval).
- Projection head: 128-d with L2 norm → compact, cosine-similarity-ready embeddings.
"""
from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class GeM(nn.Module):
    """
    Generalised Mean Pooling.

    Why: unlike average pooling which weights all spatial positions equally,
    GeM with p>1 emphasises high-activation regions (salient texture patterns).
    Standard in image retrieval (Radenovic et al., 2019).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, C)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1,
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)


class SareeEmbedder(nn.Module):
    """
    Full embedding model: backbone → GeM → projection → L2 norm.

    Output: L2-normalised embedding vector, ready for cosine similarity.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        embedding_dim: int = 128,
        gem_p: float = 3.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Create backbone without classifier (features only)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier
            global_pool="",  # remove pooling (we use GeM)
        )
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)
            if feat.dim() == 2:
                # Some timm models flatten even with global_pool=""
                feat_dim = feat.shape[1]
                self.needs_pool = False
            else:
                feat_dim = feat.shape[1]
                self.needs_pool = True

        log.info(f"Backbone {backbone_name}: feature_dim={feat_dim}, needs_pool={self.needs_pool}")

        # GeM pooling
        self.gem = GeM(p=gem_p) if self.needs_pool else nn.Identity()

        # Projection head: FC → BN → ReLU → Dropout → FC → L2norm
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → L2-normalised embedding.

        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, embedding_dim) L2-normalised embeddings
        """
        feat = self.backbone(x)  # (B, C, H, W) or (B, C)
        feat = self.gem(feat)     # (B, C)
        emb = self.projector(feat)  # (B, embedding_dim)
        emb = F.normalize(emb, p=2, dim=1)  # L2 norm
        return emb


def build_model(cfg: dict[str, Any]) -> SareeEmbedder:
    """Build model from config."""
    model_cfg = cfg.get("model", {})
    model = SareeEmbedder(
        backbone_name=model_cfg.get("backbone", "mobilenetv3_small_100"),
        pretrained=model_cfg.get("pretrained", True),
        embedding_dim=model_cfg.get("embedding_dim", 128),
        gem_p=model_cfg.get("gem_p", 3.0),
        dropout=model_cfg.get("dropout", 0.1),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    return model
