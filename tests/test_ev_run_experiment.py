import unittest
import os
import yaml
from unittest.mock import patch, MagicMock
from ev_run_experiment import load_experiment_config, prepare_experiment_params, run_experiment
import shutil

class TestEvRunExperiment(unittest.TestCase):

    def setUp(self):
        """Set up temporary configuration and environment for testing."""
        self.test_config_path = "test_experiment.yml"
        self.test_config = {
            "MPEG_TMC13_DIR": "./mpeg_tmc13",
            "PCERROR": "./pc_error",
            "MPEG_DATASET_DIR": "./mpeg_dataset",
            "EXPERIMENT_DIR": "./experiment_dir",
            "model_configs": [
                {
                    "id": "model1",
                    "config": "config1",
                    "lambdas": [1e-4, 1e-5]
                }
            ],
            "opt_metrics": ["metric1"],
            "max_deltas": [0.1],
            "data": [
                {
                    "pc_name": "test_pc",
                    "cfg_name": "test_cfg",
                    "input_pc": "input_pc.ply",
                    "input_norm": "input_norm.ply"
                }
            ]
        }

        # Create necessary directories
        for dir_path in [
            self.test_config["MPEG_DATASET_DIR"],
            self.test_config["EXPERIMENT_DIR"]
        ]:
            os.makedirs(dir_path, exist_ok=True)

        with open(self.test_config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up temporary files and directories."""
        # Remove test configuration file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        
        # Remove test directories
        for dir_path in [
            self.test_config["MPEG_DATASET_DIR"],
            self.test_config["EXPERIMENT_DIR"]
        ]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    def test_load_experiment_config(self):
        """Test loading configuration from YAML file."""
        config = load_experiment_config(self.test_config_path)
        self.assertEqual(config, self.test_config, "Configuration loading failed.")

    def test_prepare_experiment_params(self):
        """Test preparing experiment parameters."""
        config = load_experiment_config(self.test_config_path)
        params = prepare_experiment_params(config)

        expected_output_dir = os.path.join(
            config["EXPERIMENT_DIR"], "test_pc", "model1", "1.00e-04"
        )
        self.assertEqual(len(params), 2, "Incorrect number of experiment parameters prepared.")
        self.assertEqual(params[0]["output_dir"], expected_output_dir, "Output directory mismatch.")

    @patch("ev_run_experiment.Popen")
    def test_run_experiment(self, mock_popen):
        """Test running a single experiment."""
        # Configure mock
        mock_process = MagicMock()
        mock_process.returncode = 0  # Simulate successful execution
        mock_popen.return_value = mock_process

        config = load_experiment_config(self.test_config_path)
        params = prepare_experiment_params(config)[0]

        # Test successful execution
        run_experiment(params, no_stream_redirection=True)
        
        # Verify Popen was called with correct arguments
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]  # Get the command arguments
        self.assertTrue(any('--config=test_cfg' in arg for arg in args))
        self.assertTrue(any('--modelPath=model1' in arg for arg in args))

        # Test failed execution
        mock_process.returncode = 1  # Simulate failed execution
        with self.assertRaises(RuntimeError):
            run_experiment(params, no_stream_redirection=True)

if __name__ == "__main__":
    unittest.main()