import argparse
import sys
import yaml
import logging
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Load YAML configuration from file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)




def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description="Benchmark launcher")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    try:
        # Load configuration
        config = load_config(args.config)
        logging.debug(f"Loaded configuration: {config}")

        # Run benchmark
        run_benchmark(config)

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)
