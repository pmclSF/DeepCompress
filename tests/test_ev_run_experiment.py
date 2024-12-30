import sys
import os
import time
import tempfile
import pytest
import yaml

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ev_run_experiment import run_experiment

def create_dummy_experiment_file():
    """Create a dummy YAML experiment file for testing."""
    temp_dataset_dir = tempfile.mkdtemp()  # Create a temporary directory for dataset
    temp_experiment_dir = tempfile.mkdtemp()  # Create a temporary directory for experiments

    experiments = {
        'MPEG_TMC13_DIR': '/path/to/tmc13',  # This can remain as a placeholder if not used
        'PCERROR': '/path/to/pcerror',  # Placeholder if not validated
        'MPEG_DATASET_DIR': temp_dataset_dir,
        'EXPERIMENT_DIR': temp_experiment_dir,
        'pcerror_mpeg_mode': 'default',
        'model_configs': [
            {
                'id': 'model_1',
                'config': 'config_1',
                'lambdas': [0.01, 0.1],
                'opt_metrics': ['d1', 'd2'],
                'max_deltas': [1, 2],
                'fixed_threshold': True
            }
        ],
        'opt_metrics': ['d1', 'd2'],
        'max_deltas': [1, 2],
        'fixed_threshold': True,
        'data': [
            {
                'pc_name': 'test_pc',
                'cfg_name': 'cfg_1',
                'input_pc': 'input_pc.ply',
                'input_norm': 'input_norm.ply'
            }
        ]
    }
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    with open(temp_file.name, 'w') as f:
        yaml.dump(experiments, f)

    return temp_file.name, temp_dataset_dir, temp_experiment_dir


def test_argument_parsing():
    """Test if the script handles argument parsing correctly."""
    experiment_file, temp_dataset_dir, temp_experiment_dir = create_dummy_experiment_file()
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ev_run_experiment.py'))
        result = os.system(f"python {script_path} {experiment_file} --num_parallel 4 --no_stream_redirection")
        assert result == 0, "Script should execute without errors for valid input."
    finally:
        os.remove(experiment_file)
        os.rmdir(temp_dataset_dir)
        os.rmdir(temp_experiment_dir)


def test_run_experiment():
    """Test the core run_experiment functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        model_dir = os.path.join(temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        result = run_experiment(
            output_dir=output_dir,
            model_dir=model_dir,
            model_config='dummy_config',
            pc_name='test_pc',
            pcerror_path='/path/to/pcerror',
            pcerror_cfg_path='/path/to/pcerror.cfg',
            input_pc='/path/to/input.ply',
            input_norm='/path/to/norm.ply',
            opt_metrics=['d1'],
            max_deltas=[1],
            fixed_threshold=True,
            no_stream_redirection=True
        )
        assert result is not None, "run_experiment should return a valid process."


def test_load_experiment_config():
    """Test loading the YAML experiment configuration."""
    experiment_file, temp_dataset_dir, temp_experiment_dir = create_dummy_experiment_file()
    try:
        with open(experiment_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        assert 'MPEG_TMC13_DIR' in config, "Key MPEG_TMC13_DIR should exist in the configuration."
        assert 'data' in config, "Key data should exist in the configuration."
    finally:
        os.remove(experiment_file)
        os.rmdir(temp_dataset_dir)
        os.rmdir(temp_experiment_dir)

if __name__ == '__main__':
    pytest.main([__file__])
