"""Build manifest from raw data."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saree_reid.data.manifest import build_manifest
from saree_reid.utils import setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Build manifest.csv")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/manifest.csv")
    args = parser.parse_args()

    build_manifest(args.data_root, args.output)


if __name__ == "__main__":
    main()
