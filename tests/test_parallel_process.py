import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from parallel_process import parallel_process, Popen
import subprocess

# Test parallel_process function
def test_parallel_process():
    def sample_function(x, y):
        return x + y

    params = [(1, 2), (3, 4), (5, 6)]
    results = parallel_process(sample_function, params, num_parallel=2)

    # Assert results are as expected
    assert results == [3, 7, 11]

# Test Popen wrapper
def test_popen():
    command = ["echo", "Hello, World!"]
    process = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Assert the output is correct
    assert process.returncode == 0
    assert stdout.strip() == "Hello, World!"
    assert stderr == ""

# Edge case: Empty params list
def test_parallel_process_empty():
    def sample_function(x, y):
        return x + y

    params = []
    results = parallel_process(sample_function, params, num_parallel=2)

    # Assert the results are empty
    assert results == []
