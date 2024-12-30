# parallel_process.py

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, List, Tuple

def parallel_process(
    func: Callable[..., Any],
    params: List[Tuple],
    num_parallel: int = 4,
    timeout: int = None
) -> List[Any]:
    """
    Execute a function in parallel with multiple parameters using threading.

    Args:
        func (Callable[..., Any]): The function to execute in parallel.
        params (List[Tuple]): A list of tuples, where each tuple contains the arguments for a single function call.
        num_parallel (int): Number of parallel threads.
        timeout (int): Timeout in seconds for each function call (optional).

    Returns:
        List[Any]: A list of results corresponding to the function calls.
    """
    results = [None] * len(params)  # Placeholder for results to preserve order
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = {executor.submit(func, *param): idx for idx, param in enumerate(params)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=timeout)
            except Exception as e:
                print(f"Error with parameters {params[idx]}: {e}")
                results[idx] = None

    return results

def Popen(command: List[str], stdout=None, stderr=None) -> subprocess.Popen:
    """
    Wrapper for subprocess.Popen to execute a shell command.

    Args:
        command (List[str]): Command to execute.
        stdout: File or stream to capture standard output (default: None).
        stderr: File or stream to capture standard error (default: None).

    Returns:
        subprocess.Popen: The process object for the executed command.
    """
    return subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)

# Test the functionality
if __name__ == "__main__":
    def test_function(x, y):
        return x + y

    params = [(1, 2), (3, 4), (5, 6)]
    results = parallel_process(test_function, params, num_parallel=2)
    print("Parallel process results:", results)

    command = ["echo", "Hello, World!"]
    process = Popen(command)
    process.communicate()