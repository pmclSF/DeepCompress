import time
from functools import wraps

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
        print(f"{func.__name__} executed in {elapsed_time:.2f} seconds.")
        return result

    return wrapper
