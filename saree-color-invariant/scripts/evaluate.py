"""Run full evaluation: identification + verification + diagnostics."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saree_reid.data.dataset import SareeDataset
from saree_reid.data.transforms import build_eval_transform, build_recolour_transform
from saree_reid.eval.identification import extract_embeddings, run_identification
from saree_reid.eval.verification import run_verification
from saree_reid.eval.diagnostics import perturbation_stability
from saree_reid.infer import load_model_from_checkpoint
from saree_reid.utils import ensure_dir, get_device, load_config, setup_logging

import logging
log = logging.getLogger(__name__)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    # Load model
    model, cfg = load_model_from_checkpoint(args.checkpoint, cfg)
    model = model.to(device)
    model.eval()

    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg["data_root"])
    manifest_path = Path(data_cfg["manifest_path"])
    results_dir = ensure_dir(cfg.get("output", {}).get("results_dir", "results"))

    eval_transform = build_eval_transform(cfg)

    all_results = {}

    # === Identification (test split) ===
    test_ds = SareeDataset(manifest_path, data_root, split="test", transform=eval_transform)
    if len(test_ds) > 1:
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=cfg.get("eval", {}).get("gallery_batch_size", 32),
            shuffle=False, num_workers=0,
        )
        id_results = run_identification(
            model, test_loader, device,
            k_vals=cfg.get("eval", {}).get("k_vals", [1, 5, 10]),
        )
        all_results["identification_test"] = id_results
        log.info(f"Test identification: {id_results}")

    # === Verification (test pairs with synthetic recolours) ===
    recolour_transform = build_recolour_transform(cfg)
    pairs_path = Path(data_cfg.get("pairs_test_path", "data/pairs_test.csv"))
    if pairs_path.exists():
        import pandas as pd
        pairs_df = pd.read_csv(pairs_path)

        similarities = []
        labels = []
        for _, row in pairs_df.iterrows():
            img_a_path = data_root / row["path_a"]
            img_b_path = data_root / row["path_b"]

            try:
                from PIL import Image
                img_a = Image.open(img_a_path).convert("RGB")
                img_b = Image.open(img_b_path).convert("RGB")

                # For positive pairs (same image), apply recolour to one
                if row["label"] == 1:
                    tensor_a = eval_transform(img_a).unsqueeze(0).to(device)
                    tensor_b = recolour_transform(img_b).unsqueeze(0).to(device)
                else:
                    tensor_a = eval_transform(img_a).unsqueeze(0).to(device)
                    tensor_b = eval_transform(img_b).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb_a = model(tensor_a).cpu().numpy()[0]
                    emb_b = model(tensor_b).cpu().numpy()[0]

                sim = float(emb_a @ emb_b)
                similarities.append(sim)
                labels.append(row["label"])
            except Exception as e:
                log.warning(f"Skipping pair: {e}")

        if similarities:
            sims = np.array(similarities)
            labs = np.array(labels)
            ver_results = run_verification(sims, labs)
            all_results["verification_test"] = ver_results
            log.info(f"Test verification: {ver_results}")

    # === Perturbation stability ===
    if len(test_ds) > 0:
        # Take first batch for diagnostics
        batch_images = torch.stack([test_ds[i][0] for i in range(min(16, len(test_ds)))])
        stab_results = perturbation_stability(model, batch_images, device)
        all_results["perturbation_stability"] = stab_results

    # Save results
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Generate results.md
    generate_results_md(all_results, results_dir / "results.md")

    return all_results


def generate_results_md(results: dict, output_path: Path):
    """Auto-generate results.md from JSON results."""
    lines = ["# Results\n", "*Auto-generated from results.json*\n"]

    if "identification_test" in results:
        r = results["identification_test"]
        lines.append("## Identification (Test Split)\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| mAP | {r.get('mAP', 'N/A')} |")
        for k in [1, 5, 10]:
            key = f"rank_{k}"
            if key in r:
                lines.append(f"| Rank-{k} | {r[key]} |")
        lines.append(f"| # Images | {r.get('num_images', 'N/A')} |")
        lines.append(f"| # Designs | {r.get('num_designs', 'N/A')} |")
        lines.append("")

    if "verification_test" in results:
        r = results["verification_test"]
        lines.append("## Verification (Test Pairs)\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| ROC-AUC | {r.get('auc', 'N/A')} |")
        lines.append(f"| EER | {r.get('eer', 'N/A')} |")
        lines.append(f"| TAR@FAR=1e-2 | {r.get('tar_at_far_1e2', 'N/A')} |")
        lines.append(f"| TAR@FAR=1e-3 | {r.get('tar_at_far_1e3', 'N/A')} |")
        lines.append(f"| Accuracy@threshold | {r.get('accuracy_at_threshold', 'N/A')} |")
        lines.append("")

    if "perturbation_stability" in results:
        r = results["perturbation_stability"]
        lines.append("## Colour Invariance — Perturbation Stability\n")
        lines.append("| Perturbation | Mean Cosine | Std | Min |")
        lines.append("|-------------|-------------|-----|-----|")
        for ptype in ["hue_rotation", "grayscale", "channel_permutation"]:
            if ptype in r:
                p = r[ptype]
                lines.append(f"| {ptype} | {p['mean_cosine']} | {p['std_cosine']} | {p['min_cosine']} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Results table written to {output_path}")


if __name__ == "__main__":
    main()
