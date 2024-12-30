import sys
import os
import time
import pytest

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from experiment import timing

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
