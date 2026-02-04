from pathlib import Path

from training.src.cli import parse_config_args
from training.src.config import OfflineExperimentConfig
from training.src.steps.split_orders import split_orders_by_time
from training.src.pipelines.offline_experiment import run


# --------------------------------------------------
# Experiment configuration 
# --------------------------------------------------
CONFIG_PATH = Path("training/configs/offline_experiment.yaml")
CONFIG = OfflineExperimentConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else OfflineExperimentConfig()


def main():
    config_path, _ = parse_config_args(
        default_config=CONFIG_PATH,
        default_features_config=None,
        description="Run offline experiment",
    )
    config = (
        OfflineExperimentConfig.from_yaml(config_path)
        if config_path.exists()
        else OfflineExperimentConfig()
    )
    run(config)


if __name__ == "__main__":
    main()
