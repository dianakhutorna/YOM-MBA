from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple


def parse_config_args(
    *,
    default_config: Path,
    default_features_config: Path | None = None,
    description: str | None = None,
) -> Tuple[Path, Path | None]:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to the script config YAML",
    )
    if default_features_config is not None:
        parser.add_argument(
            "--features-config",
            default=str(default_features_config),
            help="Path to the feature config YAML",
        )
    args = parser.parse_args()
    features_path = None
    if default_features_config is not None:
        features_path = Path(args.features_config)
    return Path(args.config), features_path
