import os
import time
import pickle
import json
import functools
import logging
from typing import Callable, Any, List, Union, Optional
import re

# spec
#    @depends_on(Logger, Configuration, FunctionMapping)
#    interface Debug {
#        """
#        Applies the debugging process to the function.
#
#        Preconditions:
#        - `func` must be a callable.
#        - Configuration must allow debugging.
#
#        Postconditions:
#        - If debugging is allowed by the Configuration:
#          - Returns a new function that wraps the original function with debugging functionality.
#          - The returned function, when called, performs two forms of logging:
#            1. Prints function call and return information to the console, surrounded by XML tags
#               containing the callable's module path and name. The console log messages are in the
#               format `<module.function>CALL/RETURN args/result</module.function>`. For all array
#               or tensor types (i.e., objects with a .shape and/or .dtype attribute), the shapes
#               and data types are also printed.
#            2. Serializes function inputs and outputs to a log file using the `logCall` and `logReturn`
#               methods of the Logger interface. The serialized data can be loaded using the `LoadLog`
#               method. If serialization fails, the console logging still occurs, but no log file is
#               generated for that invocation.
#          - Logs only the first two invocations of the function.
#        - If debugging is not allowed by the Configuration:
#          - Returns the original function unchanged, without any debugging functionality.
#        """
#        Callable decorate(Callable func);
#    };

# Import StateCapture
from eryx.autotest.state_capture import StateCapture

## implementation
import time
import os
import pickle
import json
import inspect
from typing import Callable, Any, List, Union, Optional, Dict, Set
import re
from .configuration import Configuration
from .serializer import Serializer
from .logger import Logger
from .functionmapping import FunctionMapping

def make_invocation_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

class Debug:
    def __init__(self):
        self.configuration = Configuration()
        self.serializer = Serializer()
        self.logger = Logger()
        self.function_mapping = FunctionMapping()

    def decorate(self, func: Callable) -> Callable:
        increment_count = make_invocation_counter()
        if not self.configuration.getDebugFlag():
            return func

        else:
            module_path = self.function_mapping.get_module_path(func)
            function_name = func.__name__

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                invocation_count = increment_count()
                if invocation_count > 2:
                    return func(*args, **kwargs)
                
                log_file_path = self.function_mapping.get_log_file_path(func)
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

                # Check if this is a method call (first arg is self)
                is_method = args and hasattr(args[0], '__dict__') and not isinstance(args[0], type)
                obj = args[0] if is_method else None
                
                # Capture object state before method execution if this is a method
                if is_method:
                    class_name = obj.__class__.__name__
                    before_state_log_path = os.path.join(
                        os.path.dirname(log_file_path),
                        f"{module_path}.{class_name}._state_before_{function_name}.log"
                    )
                    try:
                        # Initialize state capturer
                        state_capturer = StateCapture(
                            max_depth=10,
                            exclude_attrs=[]
                        )
                        
                        # Capture pre-execution state
                        before_state = state_capturer.capture_state(obj)
                        self.logger.saveStateLog(before_state_log_path, before_state)
                    except Exception as e:
                        print(f"Error capturing before state: {e}")

                try:
                    serialized_args = self.serializer.serialize(args)
                    serialized_kwargs = self.serializer.serialize(kwargs)
                    self.logger.logCall(serialized_args, serialized_kwargs, log_file_path)
                except ValueError:
                    pass  # If serialization fails, just proceed with console logging

                console_log_start = f"<{module_path}.{function_name}>CALL"
                console_log_args = self._formatConsoleLog(args)
                console_log_kwargs = self._formatConsoleLog(kwargs)
                print(console_log_start)
                print(console_log_args)
                print(console_log_kwargs)

                start_time = time.time()

                result = func(*args, **kwargs)
                
                # Capture object state after method execution if this is a method
                if is_method:
                    class_name = obj.__class__.__name__
                    after_state_log_path = os.path.join(
                        os.path.dirname(log_file_path),
                        f"{module_path}.{class_name}._state_after_{function_name}.log"
                    )
                    try:
                        # Capture post-execution state using the same state capturer
                        after_state = state_capturer.capture_state(obj)
                        self.logger.saveStateLog(after_state_log_path, after_state)
                    except Exception as e:
                        print(f"Error capturing after state: {e}")
                
                try:
                    serialized_result = self.serializer.serialize(result)
                    self.logger.logReturn(serialized_result, time.time() - start_time, log_file_path)

                    console_log_end = f"</{module_path}.{function_name}>RETURN"
                    console_log_result = self._formatConsoleLog(result)
                    print(console_log_end + " " + console_log_result)

                except Exception as e:
                    self.logger.logError(str(e), log_file_path)
                    print(f"<{module_path}.{function_name}>ERROR {str(e)}")
                return result

            return wrapper

    def _formatConsoleLog(self, data: Any) -> str:
        if not isinstance(data, tuple):
            data = (data,)

        formatted_data = []
        for item in data:
            if hasattr(item, 'shape') and hasattr(item, 'dtype'):
                formatted_data.append(f"type={type(item)}, shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, (int, float, str, bool)):
                formatted_data.append(f"type={type(item)}, {item}")
            else:
                formatted_data.append(f"type={type(item)}")
        return ", ".join(formatted_data)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

import unittest

class TestDebug(unittest.TestCase):
    def setUp(self):
        self.configuration = Configuration()
        self.serializer = Serializer()
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.debug = Debug(self.configuration, self.serializer, self.logger, self.function_mapping)

    def test_decorate_call(self):
        @self.debug.decorate
        def add(x, y):
            return x + y

        result = add(3, 4)
        self.assertEqual(result, 7)

    def test_decorate_return(self):
        @self.debug.decorate
        def multiply(x, y):
            return x * y

        result = multiply(2, 3)
        self.assertEqual(result, 6)
        result = multiply(4, 5)
        self.assertEqual(result, 20)
        result = multiply(6, 7)  # This call should not be logged
        self.assertEqual(result, 42)

    def test_decorate_error(self):
        @self.debug.decorate
        def divide(x, y):
            return x / y

        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)

#    def test_format_console_log(self):
#        data = (3, "hello")
#        formatted_log = self.debug._formatConsoleLog(data)
#        self.assertEqual(formatted_log, "3, hello")

# Import the global configuration
from eryx.autotest_config import config

# Create a global instance of Debug
def _create_debug_decorator():
    debug_obj = Debug()
    # Use the configuration from autotest_config
    debug_obj.configuration = config
    return debug_obj.decorate

# Create the decorator based on the configuration
if config.getDebugFlag():
    # Create the enhanced debug decorator with state capture support
    def debug(func: Optional[Callable] = None, 
              capture_state: bool = True,
              max_depth: int = 10,
              exclude_attrs: Optional[List[str]] = None) -> Callable:
        """
        Enhanced debug decorator with state capture support.
        
        Args:
            func: Function to decorate
            capture_state: Whether to capture object state for methods
            max_depth: Maximum recursion depth for state capture
            exclude_attrs: List of attribute patterns to exclude from capture
            
        Returns:
            Decorated function with debug and state capture functionality
        """
        debug_obj = Debug()
        debug_obj.configuration = config
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get logger and function mapping
                logger = Logger()
                function_mapping = FunctionMapping()
                
                # Determine module path and function name
                module_path = function_mapping.get_module_path(func)
                function_name = func.__name__
                
                # Check if this is a method call (first arg is self/cls)
                is_method = False
                obj = None
                class_name = None
                if args and hasattr(args[0], '__dict__') and not isinstance(args[0], type):
                    is_method = True
                    obj = args[0]
                    class_name = obj.__class__.__name__
                
                # Create log paths
                log_file_path = function_mapping.get_log_file_path(func)
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                
                # Create consistent state log paths - IMPORTANT CHANGE
                # Use format: {module_path}.{class_name}._state_{before/after}_{function_name}.log
                state_capturer = None
                if capture_state and is_method:
                    before_state_path = os.path.join(
                        os.path.dirname(log_file_path),
                        f"{module_path}.{class_name}._state_before_{function_name}.log"
                    )
                    
                    try:
                        # Initialize state capturer with enhanced serializer
                        state_capturer = StateCapture(
                            max_depth=max_depth,
                            exclude_attrs=exclude_attrs
                        )
                        
                        # Capture and save before state
                        before_state = state_capturer.capture_state(obj)
                        logger.saveStateLog(before_state_path, before_state)
                        print(f"Captured before state: {before_state_path}")
                    except Exception as e:
                        print(f"Error capturing before state: {str(e)}")
                
                # Log function call
                try:
                    serialized_args = logger.serializer.serialize(args)
                    serialized_kwargs = logger.serializer.serialize(kwargs)
                    logger.logCall(serialized_args, serialized_kwargs, log_file_path)
                except Exception as e:
                    print(f"Error logging function call: {str(e)}")
                
                # Call original function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log function return
                try:
                    serialized_result = logger.serializer.serialize(result)
                    logger.logReturn(serialized_result, execution_time, log_file_path)
                except Exception as e:
                    print(f"Error logging function return: {str(e)}")
                
                # Capture state after method execution
                if capture_state and is_method and state_capturer:
                    after_state_path = os.path.join(
                        os.path.dirname(log_file_path),
                        f"{module_path}.{class_name}._state_after_{function_name}.log"
                    )
                    
                    try:
                        # Capture and save after state
                        after_state = state_capturer.capture_state(obj)
                        logger.saveStateLog(after_state_path, after_state)
                        print(f"Captured after state: {after_state_path}")
                    except Exception as e:
                        print(f"Error capturing after state: {str(e)}")
                
                return result
            
            return wrapper
        
        # Handle both @debug and @debug(config)
        if func is None:
            return decorator
        return decorator(func)
else:
    # Provide a no-op decorator when debugging is disabled
    def debug(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

if __name__ == '__main__':
    import unittest
    unittest.main(argv=[''], verbosity=2, exit=False)

