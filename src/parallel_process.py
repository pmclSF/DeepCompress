import multiprocessing
import subprocess
import logging
import time
from typing import Callable, Any, List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from queue import Queue

@dataclass
class ProcessResult:
    """Container for process execution results."""
    index: int
    result: Any
    success: bool
    error: Optional[Exception] = None

class ProcessTimeoutError(Exception):
    """Custom exception for process timeouts."""
    pass

class Popen:
    def __init__(self, 
                 cmd: List[str], 
                 stdout: Any = None, 
                 stderr: Any = None,
                 timeout: Optional[float] = None):
        """
        Enhanced subprocess management with timeout support.
        """
        self.cmd = cmd
        self.timeout = timeout
        self.start_time = time.time()
        
        self.process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True
        )

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for the subprocess to complete with timeout."""
        wait_timeout = timeout or self.timeout
        
        if wait_timeout is not None:
            try:
                return self.process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                self.terminate()
                raise ProcessTimeoutError(
                    f"Process timed out after {wait_timeout} seconds: {' '.join(self.cmd)}"
                )
        
        return self.process.wait()

    def terminate(self):
        """Terminate the subprocess with cleanup."""
        try:
            self.process.terminate()
            self.process.wait(timeout=1.0)
        except (subprocess.TimeoutExpired, ProcessTimeoutError):
            self.process.kill()
        finally:
            if hasattr(self.process, 'stdout') and self.process.stdout:
                self.process.stdout.close()
            if hasattr(self.process, 'stderr') and self.process.stderr:
                self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

def parallel_process(
    func: Callable[[Any], Any],
    params_list: List[Any],
    num_parallel: int = 4,
    max_retries: int = 3,
    timeout: Optional[float] = None
) -> List[Any]:
    """
    Enhanced parallel execution with retries and result ordering.
    """
    def wrapped_func(param: Any) -> Any:
        """Wrapper to handle timeouts at the function level."""
        if timeout:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, param)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError:
                    raise ProcessTimeoutError(f"Task timed out after {timeout} seconds")
        return func(param)

    def worker(index: int, param: Any, result_queue: Queue) -> None:
        """Worker function that handles retries."""
        for attempt in range(max_retries):
            try:
                result = wrapped_func(param)
                result_queue.put(ProcessResult(
                    index=index,
                    result=result,
                    success=True
                ))
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Task failed after {max_retries} attempts: {str(e)}")
                    result_queue.put(ProcessResult(
                        index=index,
                        result=None,
                        success=False,
                        error=e
                    ))
                else:
                    logging.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.1)  # Small delay between retries

    # Process all parameters in parallel
    result_queue: Queue = Queue()
    results_dict: Dict[int, ProcessResult] = {}
    
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [
            executor.submit(worker, i, param, result_queue)
            for i, param in enumerate(params_list)
        ]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will propagate any exceptions
    
    # Collect results and maintain order
    while len(results_dict) < len(params_list):
        result = result_queue.get()
        results_dict[result.index] = result
    
    # Process results in order
    ordered_results = []
    for i in range(len(params_list)):
        result = results_dict[i]
        if not result.success:
            if isinstance(result.error, ProcessTimeoutError):
                raise TimeoutError(str(result.error))
            raise result.error or RuntimeError(f"Task {i} failed without specific error")
        ordered_results.append(result.result)
        
    return ordered_results