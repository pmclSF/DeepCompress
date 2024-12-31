import yaml
import os
import logging
from typing import Dict, Any

class ExperimentConfig:
    def __init__(self, config_path: str):
        """
        Load and validate experiment configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Failed to load configuration file: {e}")
            raise

    def _validate_config(self):
        """Validate the experiment configuration."""
        required_keys = ["dataset_path", "experiment_dir", "model_configs", "metrics"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate paths
        if not os.path.exists(self.config["dataset_path"]):
            raise ValueError(f"Dataset path does not exist: {self.config['dataset_path']}")

        os.makedirs(self.config["experiment_dir"], exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the configuration."""
        return self.config.get(key, default)

class Experiment:
    def __init__(self, config_path: str):
        """
        Initialize an experiment based on a configuration file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = ExperimentConfig(config_path)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for the experiment."""
        logger = logging.getLogger("Experiment")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        logger.addHandler(handler)
        return logger

    def run(self):
        """Run the experiment pipeline."""
        self.logger.info("Starting experiment.")

        # Example: Iterate through model configurations
        model_configs = self.config.get("model_configs")
        for model_config in model_configs:
            self.logger.info(f"Running model configuration: {model_config}")
            # Placeholder for model training or evaluation
            self._run_model_pipeline(model_config)

        self.logger.info("Experiment completed.")

    def _run_model_pipeline(self, model_config: Dict[str, Any]):
        """Run a single model pipeline based on the configuration."""
        # Placeholder for actual training/evaluation logic
        self.logger.info(f"Processing configuration: {model_config}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an experiment based on a configuration file.")
    parser.add_argument("config", type=str, help="Path to the experiment configuration YAML file.")

    args = parser.parse_args()

    experiment = Experiment(args.config)
    experiment.run()
