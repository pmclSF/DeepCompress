import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from experiment import Experiment, ExperimentConfig


class TestExperimentConfig(unittest.TestCase):
    def setUp(self):
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
        for path in [self.test_config_path,
                    self.test_config["dataset_path"],
                    self.test_config["experiment_dir"]]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    os.rmdir(path)
                else:
                    os.remove(path)

    def test_load_config(self):
        config = ExperimentConfig(self.test_config_path)
        self.assertEqual(config.get("dataset_path"), self.test_config["dataset_path"])
        self.assertEqual(config.get("experiment_dir"), self.test_config["experiment_dir"])
        self.assertEqual(config.get("model_configs"), self.test_config["model_configs"])
        self.assertEqual(config.get("metrics"), self.test_config["metrics"])

    def test_validate_config(self):
        config = ExperimentConfig(self.test_config_path)
        self.assertTrue(os.path.exists(config.get("dataset_path")))
        self.assertTrue(os.path.exists(config.get("experiment_dir")))

class TestExperiment(unittest.TestCase):
    def setUp(self):
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
        for path in [self.test_config_path,
                    self.test_config["dataset_path"],
                    self.test_config["experiment_dir"]]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    os.rmdir(path)
                else:
                    os.remove(path)

    def test_run_experiment(self):
        self.experiment.run()
        self.assertTrue(os.path.exists(self.test_config["experiment_dir"]))

if __name__ == "__main__":
    unittest.main()
