from __future__ import annotations

from pathlib import Path

from training.src.cli import parse_config_args
from training.src.pipelines.training_experiment import ExperimentPipelineConfig, run


DEFAULT_CONFIG_PATH = Path("training/configs/experiment_1m.yaml")


def main() -> None:
    config_path, _ = parse_config_args(
        default_config=DEFAULT_CONFIG_PATH,
        default_features_config=None,
        description="Run experiment training pipeline (separate train/test files)",
    )
    config = ExperimentPipelineConfig.from_yaml(config_path)
    run(config)


if __name__ == "__main__":
    main()
