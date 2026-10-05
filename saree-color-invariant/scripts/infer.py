"""Inference CLI: identify or verify."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saree_reid.data.transforms import build_eval_transform
from saree_reid.infer import identify, load_model_from_checkpoint, verify
from saree_reid.utils import get_device, load_config, setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Inference")
    sub = parser.add_subparsers(dest="mode")

    # Identify mode
    id_parser = sub.add_parser("identify")
    id_parser.add_argument("--query", type=str, required=True)
    id_parser.add_argument("--gallery-dir", type=str, required=True)
    id_parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    id_parser.add_argument("--config", type=str, default="configs/default.yaml")
    id_parser.add_argument("--top-k", type=int, default=5)

    # Verify mode
    ver_parser = sub.add_parser("verify")
    ver_parser.add_argument("--image-a", type=str, required=True)
    ver_parser.add_argument("--image-b", type=str, required=True)
    ver_parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    ver_parser.add_argument("--config", type=str, default="configs/default.yaml")
    ver_parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    model, cfg = load_model_from_checkpoint(args.checkpoint, cfg)
    model = model.to(device)
    transform = build_eval_transform(cfg)

    if args.mode == "identify":
        gallery_dir = Path(args.gallery_dir)
        gallery_paths = sorted(
            p for p in gallery_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        results = identify(model, args.query, gallery_paths, transform, device, args.top_k)
        print(json.dumps(results, indent=2))

    elif args.mode == "verify":
        result = verify(model, args.image_a, args.image_b, transform, device, args.threshold)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
