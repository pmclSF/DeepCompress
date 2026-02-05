import yaml
import os
import logging
from typing import Dict, Any

class ExperimentConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def _validate_config(self):
        required_keys = ["dataset_path", "experiment_dir", "model_configs", "metrics"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        if not os.path.exists(self.config["dataset_path"]):
            raise ValueError(f"Dataset path does not exist: {self.config['dataset_path']}")

        os.makedirs(self.config["experiment_dir"], exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

class Experiment:
    def __init__(self, config_path: str):
        self.config = ExperimentConfig(config_path)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("Experiment")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def run(self):
        self.logger.info("Starting experiment.")
        model_configs = self.config.get("model_configs")
        for model_config in model_configs:
            self.logger.info(f"Running model configuration: {model_config}")
            self._run_model_pipeline(model_config)
        self.logger.info("Experiment completed.")

    def _run_model_pipeline(self, model_config: Dict[str, Any]):
        self.logger.info(f"Processing configuration: {model_config}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("config", type=str, help="Path to the experiment configuration YAML file.")
    args = parser.parse_args()
    experiment = Experiment(args.config)
    experiment.run()