#!/usr/bin/env python3
"""
main.py — Launch a benchmark using parameters from a flat YAML config file.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Any, Dict

import yaml  # pip install pyyaml


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a flat YAML config file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_benchmark(params: Dict[str, Any]) -> None:
    """Simulated benchmark using provided parameters."""
    logging.info("=== Running benchmark ===")
    for key, value in params.items():
        logging.info("%s : %s", key, value)

    import time, random
    random.seed(params["seed"])
    for i in range(1, params["iterations"] + 1):
        logging.info("Iteration %d/%d ...", i, params["iterations"])
        time.sleep(params["duration_s"] / max(1, params["iterations"]))
    logging.info("Benchmark finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark launcher (YAML config)")
    parser.add_argument(
        "-c", "--config",
        required=True,
        type=Path,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    try:
        cfg = load_yaml_config(args.config)
        logging.debug("Loaded config: %s", cfg)

        # --- Extract parameters directly ---
        # Because it's flat, just use dict lookups:
        name         = cfg.get("name", "default-bench")
        iterations   = int(cfg.get("iterations", 1))
        duration_s   = float(cfg.get("duration_s", 1.0))
        dataset_path = Path(cfg.get("dataset_path", "."))
        num_workers  = int(cfg.get("num_workers", 1))
        seed         = int(cfg.get("seed", 0))

        # --- Pack into a params dict ---
        params = {
            "name": name,
            "iterations": iterations,
            "duration_s": duration_s,
            "dataset_path": dataset_path,
            "num_workers": num_workers,
            "seed": seed,
        }

        run_benchmark(params)

    except Exception as e:
        logging.error("Fatal error: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
