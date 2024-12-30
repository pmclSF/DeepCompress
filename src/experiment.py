import os
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def timing(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func (callable): Function to be timed.

    Returns:
        callable: Wrapped function with timing enabled.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds.")
        return result

    return wrapper

@timing
def assert_exists(path):
    """
    Assert that the given path exists. Raises an error if it doesn't.
    
    Args:
        path (str): Path to check.
    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(path):
        logger.error(f"Required path does not exist: {path}")
        raise FileNotFoundError(f"Path does not exist: {path}")
    logger.info(f"Validated existence of path: {path}")

@timing
def validate_experiment_params(params, required_keys):
    """
    Validate that the experiment parameters contain all required keys.
    
    Args:
        params (dict): Experiment parameters to validate.
        required_keys (list): List of keys that must be present in params.
    Raises:
        ValueError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        logger.error(f"Missing required experiment parameters: {missing_keys}")
        raise ValueError(f"Missing required experiment parameters: {missing_keys}")
    logger.info("All required experiment parameters are present.")

@timing
def prepare_experiment_dir(base_dir, experiment_name):
    """
    Prepare the directory for a specific experiment, ensuring it exists.
    
    Args:
        base_dir (str): Base directory for experiments.
        experiment_name (str): Name of the experiment.
    Returns:
        str: Full path to the experiment directory.
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Prepared experiment directory: {experiment_dir}")
    return experiment_dir