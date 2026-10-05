"""
Efficiency profiling: params, FLOPs, latency, memory, throughput.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from saree_reid.models.backbone import build_model
from saree_reid.utils import get_device

log = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224),
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict[str, float]:
    """Measure inference latency with warmup."""
    model.eval()
    model.to(device)
    x = torch.randn(*input_size).to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = sorted(times)
    return {
        "median_ms": round(times[len(times)//2] * 1000, 2),
        "mean_ms": round(sum(times)/len(times) * 1000, 2),
        "min_ms": round(times[0] * 1000, 2),
        "batch_size": input_size[0],
        "device": str(device),
    }


def compute_embedding_storage(
    embedding_dim: int,
    n_gallery: int = 1_000_000,
) -> dict[str, str]:
    """Estimate gallery storage for N images."""
    fp32_bytes = n_gallery * embedding_dim * 4
    fp16_bytes = n_gallery * embedding_dim * 2
    return {
        "embedding_dim": embedding_dim,
        "n_gallery": n_gallery,
        "fp32_mb": round(fp32_bytes / 1e6, 1),
        "fp16_mb": round(fp16_bytes / 1e6, 1),
    }


def run_efficiency_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """Full efficiency report."""
    device = get_device()
    model = build_model(cfg).to(device)
    model.eval()

    params = count_parameters(model)
    embedding_dim = cfg.get("model", {}).get("embedding_dim", 128)
    input_size = cfg.get("training", {}).get("input_size", 224)

    # Latency: batch=1
    lat_b1 = measure_latency(model, device, (1, 3, input_size, input_size))

    # Latency: batch=32
    lat_b32 = measure_latency(model, device, (32, 3, input_size, input_size))

    # Storage
    storage = compute_embedding_storage(embedding_dim)

    # Try FLOPs (requires fvcore)
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, 3, input_size, input_size).to(device)
        fca = FlopCountAnalysis(model, x)
        flops = round(fca.total() / 1e6, 1)  # MFLOPs
    except ImportError:
        log.warning("fvcore not installed, skipping FLOPs")
    except Exception as e:
        log.warning(f"FLOPs computation failed: {e}")

    report = {
        "parameters": params,
        "flops_mflops": flops,
        "latency_batch1": lat_b1,
        "latency_batch32": lat_b32,
        "embedding_storage_1M": storage,
        "input_resolution": input_size,
        "backbone": cfg.get("model", {}).get("backbone", "unknown"),
    }

    log.info(f"Efficiency: {params['total']:,} params, "
             f"{flops or '?'} MFLOPs, "
             f"{lat_b1['median_ms']:.1f}ms/image (batch=1)")
    return report
