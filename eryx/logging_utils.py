"""Logging utilities for the eryx package.

This module provides decorators and utilities for consistent logging across
the codebase, including method call timing, property access logging, and
array shape tracking.
"""

import logging
import time
import functools
import numpy as np
from contextlib import contextmanager

# Configure default logging format
DEFAULT_FORMAT = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.DEBUG,
    format=DEFAULT_FORMAT,
    datefmt=DEFAULT_DATE_FORMAT,
)

def log_method_call(func):
    """Decorator to log method entry/exit with timing.
    
    Args:
        func: The function to be decorated
        
    Returns:
        wrapper: The decorated function that includes logging
        
    Example:
        @log_method_call
        def my_method(self):
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else ''
        method_name = func.__name__
        logging.debug(f"[{class_name}.{method_name}] Enter with args={args[1:]}, kwargs={kwargs}")
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.debug(f"[{class_name}.{method_name}] Exit. Duration={duration:.2f}s")
        if result is not None:
            import numpy as np
            if isinstance(result, np.ndarray):
                logging.debug(f"[{class_name}.{method_name}] Return array shape: {result.shape}, first few values: {result.flatten()[:5]}")
            else:
                logging.debug(f"[{class_name}.{method_name}] Return value: {result}")
        return result
    return wrapper

@contextmanager
def timed_operation(operation_name: str):
    """Context manager for timing a code block.
    
    Args:
        operation_name: Name of the operation being timed
        
    Yields:
        None: The context manager yields nothing
        
    Example:
        with timed_operation("my_operation"):
            # Code to time
            pass
    """
    start_time = time.time()
    yield
    duration = time.time() - start_time
    logging.debug(f"[TimedOperation:{operation_name}] Duration={duration:.2f}s")

def log_property_access(func):
    """Decorator to log property access.
    
    Args:
        func: The property function to be decorated
        
    Returns:
        wrapper: The decorated property that includes logging
        
    Example:
        @property
        @log_property_access
        def my_property(self):
            return self._value
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else ''
        prop_name = func.__name__
        value = func(*args, **kwargs)
        logging.debug(f"[{class_name}.{prop_name}] Accessed. Value={value}")
        return value
    return wrapper

def log_array_shape(array: "np.ndarray", array_name: str = "array") -> None:
    """Utility to log an array's shape.
    
    Args:
        array: NumPy array to log the shape of
        array_name: Name to use when logging the array
        
    Returns:
        None
    """
    logging.debug(f"[Array Shape] {array_name}.shape={array.shape}")
TimedOperation = timed_operation
