"""Assign splits and generate verification pairs."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saree_reid.data.splits import run_splits
from saree_reid.utils import setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Assign splits + generate pairs")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output", type=str, default="data/manifest.csv")
    parser.add_argument("--pairs-val", type=str, default="data/pairs_val.csv")
    parser.add_argument("--pairs-test", type=str, default="data/pairs_test.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_splits(args.manifest, args.output, args.pairs_val, args.pairs_test, args.seed)


if __name__ == "__main__":
    main()
