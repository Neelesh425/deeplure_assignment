"""Train the embedding model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saree_reid.train import train
from saree_reid.utils import load_config, setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Train saree embedding model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
