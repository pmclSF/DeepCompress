import sys
import os
import time
import tempfile
import pytest

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.experiment import timing, assert_exists, validate_experiment_params, prepare_experiment_dir

def slow_function(duration):
    time.sleep(duration)

@timing
def decorated_slow_function(duration):
    time.sleep(duration)

def test_timing_decorator():
    # Test the function without decorator
    start_time = time.time()
    slow_function(1)
    elapsed_time = time.time() - start_time
    assert elapsed_time >= 1, "Function did not run for the expected duration"

    # Test the decorated function
    start_time = time.time()
    decorated_slow_function(1)
    elapsed_time = time.time() - start_time
    assert elapsed_time >= 1, "Decorated function did not run for the expected duration"

    # Validate that the decorator doesn't modify the return value
    assert decorated_slow_function(0) is None, "Decorator altered the function output"

def test_assert_exists():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with an existing directory
        assert_exists(temp_dir)

        # Test with a non-existing directory
        non_existing_dir = os.path.join(temp_dir, 'non_existing')
        with pytest.raises(FileNotFoundError):
            assert_exists(non_existing_dir)

def test_validate_experiment_params():
    params = {'key1': 'value1', 'key2': 'value2'}
    required_keys = ['key1', 'key2']

    # Test with all required keys present
    validate_experiment_params(params, required_keys)

    # Test with missing keys
    missing_keys = ['key1', 'key3']
    with pytest.raises(ValueError):
        validate_experiment_params(params, missing_keys)

def test_prepare_experiment_dir():
    with tempfile.TemporaryDirectory() as base_dir:
        experiment_name = 'test_experiment'

        # Test creating an experiment directory
        experiment_dir = prepare_experiment_dir(base_dir, experiment_name)
        assert os.path.exists(experiment_dir), "Experiment directory was not created"