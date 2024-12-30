import unittest
import os
import yaml
from experiment import Experiment, ExperimentConfig

class TestExperimentConfig(unittest.TestCase):

    def setUp(self):
        """Set up a temporary configuration file for testing."""
        self.test_config_path = "test_config.yml"
        self.test_config = {
            "dataset_path": "./test_dataset",
            "experiment_dir": "./test_experiment",
            "model_configs": [
                {"model": "model1", "params": {"lr": 0.001}},
                {"model": "model2", "params": {"lr": 0.01}}
            ],
            "metrics": ["accuracy", "loss"]
        }

        os.makedirs(self.test_config["dataset_path"], exist_ok=True)
        with open(self.test_config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up the temporary files and directories."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.test_config["dataset_path"]):
            os.rmdir(self.test_config["dataset_path"])
        if os.path.exists(self.test_config["experiment_dir"]):
            os.rmdir(self.test_config["experiment_dir"])

    def test_load_config(self):
        """Test loading configuration from YAML file."""
        config = ExperimentConfig(self.test_config_path)
        self.assertEqual(config.get("dataset_path"), self.test_config["dataset_path"])
        self.assertEqual(config.get("experiment_dir"), self.test_config["experiment_dir"])
        self.assertEqual(config.get("model_configs"), self.test_config["model_configs"])
        self.assertEqual(config.get("metrics"), self.test_config["metrics"])

    def test_validate_config(self):
        """Test configuration validation."""
        config = ExperimentConfig(self.test_config_path)
        self.assertTrue(os.path.exists(config.get("dataset_path")))
        self.assertTrue(os.path.exists(config.get("experiment_dir")))

class TestExperiment(unittest.TestCase):

    def setUp(self):
        """Set up a temporary configuration file and experiment."""
        self.test_config_path = "test_config.yml"
        self.test_config = {
            "dataset_path": "./test_dataset",
            "experiment_dir": "./test_experiment",
            "model_configs": [
                {"model": "model1", "params": {"lr": 0.001}},
                {"model": "model2", "params": {"lr": 0.01}}
            ],
            "metrics": ["accuracy", "loss"]
        }

        os.makedirs(self.test_config["dataset_path"], exist_ok=True)
        with open(self.test_config_path, "w") as f:
            yaml.dump(self.test_config, f)

        self.experiment = Experiment(self.test_config_path)

    def tearDown(self):
        """Clean up the temporary files and directories."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.test_config["dataset_path"]):
            os.rmdir(self.test_config["dataset_path"])
        if os.path.exists(self.test_config["experiment_dir"]):
            os.rmdir(self.test_config["experiment_dir"])

    def test_run_experiment(self):
        """Test running the experiment pipeline."""
        self.experiment.run()
        self.assertTrue(os.path.exists(self.test_config["experiment_dir"]), "Experiment directory should exist.")

if __name__ == "__main__":
    unittest.main()
