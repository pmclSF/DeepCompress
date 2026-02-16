import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
import time
import subprocess
from unittest.mock import patch, MagicMock
from concurrent.futures import TimeoutError
from parallel_process import (
    parallel_process, 
    Popen, 
    ProcessTimeoutError,
    ProcessResult
)

def square(x):
    """Simple square function for testing."""
    return x * x

def slow_function(x):
    """Function that takes time to complete."""
    time.sleep(x)
    return x * x

def failing_function(x):
    """Function that fails on certain inputs."""
    if x == 2:  # Fail on input 2
        raise ValueError("Failing on purpose")
    return x * x

class TestParallelProcess(unittest.TestCase):
    def test_parallel_process_basic(self):
        """Test basic parallel execution with correct ordering."""
        params = [1, 2, 3, 4, 5]
        expected_results = [1, 4, 9, 16, 25]

        results = parallel_process(square, params, num_parallel=2)
        self.assertEqual(results, expected_results)

    def test_parallel_process_timeout(self):
        """Test timeout functionality."""
        params = [0.1, 0.1, 2.0]  # Last task will timeout
        
        with self.assertRaises(TimeoutError):
            parallel_process(slow_function, params, 
                           num_parallel=2, timeout=1.0)

    def test_parallel_process_retries(self):
        """Test retry functionality for failing tasks."""
        params = [1, 2, 3]  # 2 will fail
        
        with self.assertRaises(ValueError):
            parallel_process(failing_function, params, 
                           num_parallel=2, max_retries=2)

    @patch("subprocess.Popen")
    def test_popen_timeout(self, mock_popen):
        """Test Popen subprocess timeout."""
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=1.0)
        mock_process.stdout = None
        mock_process.stderr = None
        mock_popen.return_value = mock_process

        cmd = ["sleep", "10"]
        with self.assertRaises(ProcessTimeoutError):
            with Popen(cmd, timeout=1.0) as process:
                process.wait()

    @patch("subprocess.Popen")
    def test_popen_cleanup(self, mock_popen):
        """Test proper cleanup of Popen resources."""
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process

        cmd = ["echo", "test"]
        with Popen(cmd) as process:
            pass  # Context manager should handle cleanup

        mock_process.terminate.assert_called_once()
        mock_process.stdout.close.assert_called_once()
        mock_process.stderr.close.assert_called_once()

    def test_process_result_dataclass(self):
        """Test ProcessResult dataclass functionality."""
        result = ProcessResult(
            index=0,
            result=42,
            success=True,
            error=None
        )
        
        self.assertEqual(result.index, 0)
        self.assertEqual(result.result, 42)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

if __name__ == "__main__":
    unittest.main()