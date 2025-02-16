import logging
import time
import functools
from contextlib import contextmanager

# Configure logging with the required format
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_method_call(func):
    """Decorator to log method entry and exit with duration."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else ''
        method_name = func.__name__
        logging.debug(f"[{class_name}.{method_name}] Enter with args={args[1:]}, kwargs={kwargs}")
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.debug(f"[{class_name}.{method_name}] Exit. Duration={duration:.2f}s")
        return result
    return wrapper

@contextmanager
def TimedOperation(operation_name):
    """Context manager for timing a code block."""
    start_time = time.time()
    yield
    duration = time.time() - start_time
    logging.debug(f"[TimedOperation:{operation_name}] Duration={duration:.2f}s")

def log_property_access(func):
    """Decorator to log property access."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else ''
        prop_name = func.__name__
        value = func(*args, **kwargs)
        logging.debug(f"[{class_name}.{prop_name}] Accessed. Value={value}")
        return value
    return wrapper

def log_array_shape(array, array_name="array"):
    """Utility to log an array's shape."""
    logging.debug(f"[Array Shape] {array_name}.shape={array.shape}")
