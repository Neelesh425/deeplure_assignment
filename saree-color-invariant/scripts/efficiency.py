"""Run efficiency profiling."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saree_reid.efficiency import run_efficiency_report
from saree_reid.utils import ensure_dir, load_config, setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Efficiency report")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_efficiency_report(cfg)

    results_dir = ensure_dir(cfg.get("output", {}).get("results_dir", "results"))
    with open(results_dir / "efficiency.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
