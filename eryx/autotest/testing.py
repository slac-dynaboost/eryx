from .logger import Logger
from .functionmapping import FunctionMapping
from .configuration import Configuration
import unittest

from typing import List, Tuple, Any, Optional, Callable, Union

class TestSummary:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def increment_passed(self):
        self.passed += 1

    def increment_failed(self):
        self.failed += 1

    def increment_skipped(self):
        self.skipped += 1

    def __repr__(self):
        return f"TestSummary(passed={self.passed}, failed={self.failed}, skipped={self.skipped})"

class Testing:
    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
        self.logger = logger
        self.function_mapping = function_mapping

class Testing:
    def __init__(self, logger: Logger, function_mapping: FunctionMapping):
        self.logger = logger
        self.function_mapping = function_mapping

    def testCallable(self, log_path_prefix: str, func: Callable) -> bool:
        print(f"Debug: testCallable called with log_path_prefix: {log_path_prefix}")
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        print(f"Debug: Found log files: {log_files}")
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            #print(f"Debug: Loaded logs: {logs}")
            for i in range(len(logs) // 2):
                args = logs[2 * i]['args']
                kwargs = logs[2 * i]['kwargs']
                expected_output = logs[2 * i + 1]['result']
                try:
                    deserialized_args = self.logger.serializer.deserialize(args)
                    deserialized_kwargs = self.logger.serializer.deserialize(kwargs)
                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
                    actual_output = func(*deserialized_args, **deserialized_kwargs)
                    #print(f"Debug: Actual output: {actual_output}")
                    if actual_output != deserialized_expected_output:
                        print("Debug: Test failed")
                        return False
                except Exception as e:
                    print(f"Error testing function: {e}")
                    return False
        print("Debug: Test passed")
        return True

    def createTestCase(self, log_path_prefix: str) -> Union[tuple, None]:
        print(f"Debug: createTestCase called with log_path_prefix: {log_path_prefix}")
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        print(f"Debug: Found log files: {log_files}")
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            #print(f"Debug: Loaded logs: {logs}")
            if logs:
                log = logs[0]
                inputs = log['args']
                expected_output = log['result']
                func = self.function_mapping.load_function(log_file)
                print(f"Debug: Loaded function: {func}")
                if func is not None:
                    return (inputs, expected_output, func)
        print("Debug: No test case found")
        return None

    def runTestSuite(self, log_path_prefix: str) -> TestSummary:
        print(f"Debug: runTestSuite called with log_path_prefix: {log_path_prefix}")
        summary = TestSummary()
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        print(f"Debug: Found log files: {log_files}")
        for log_file in log_files:
            test_case = self.createTestCase(log_path_prefix)
            if test_case is not None:
                inputs, expected_output, func = test_case
                if self.testCallable(log_path_prefix, func):
                    summary.increment_passed()
                else:
                    summary.increment_failed()
            else:
                summary.increment_skipped()
        print(f"Debug: Test summary: {summary}")
        return summary

class TestSummary:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def increment_passed(self):
        self.passed += 1

    def increment_failed(self):
        self.failed += 1

    def increment_skipped(self):
        self.skipped += 1

    def __repr__(self):
        return f"TestSummary(passed={self.passed}, failed={self.failed}, skipped={self.skipped})"


def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y

class TestTesting(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.testing = Testing(self.logger, self.function_mapping)

    def test_testCallable(self):
        log_path_prefix = 'test_logs'
        self.logger.logReturn(log_path_prefix + '/add', (3, 4), 7)
        self.assertTrue(self.testing.testCallable(log_path_prefix, add))

    def test_createTestCase(self):
        log_path_prefix = 'test_logs'
        self.logger.logReturn(log_path_prefix + '/add', (3, 4), 7)
        self.function_mapping.save_function(log_path_prefix + '/add', add)
        test_case = self.testing.createTestCase(log_path_prefix)
        self.assertIsNotNone(test_case)
        inputs, expected_output, func = test_case
        self.assertEqual(self.logger.serializer.deserialize(inputs), (3, 4))
        self.assertEqual(self.logger.serializer.deserialize(expected_output), 7)
        self.assertEqual(func, add)

    def test_runTestSuite(self):
        log_path_prefix = 'test_logs'
        self.logger.logReturn(log_path_prefix + '/add', (3, 4), 7)
        self.logger.logReturn(log_path_prefix + '/multiply', (3, 4), 12)
        self.function_mapping.save_function(log_path_prefix + '/add', add)
        self.function_mapping.save_function(log_path_prefix + '/multiply', multiply)
        summary = self.testing.runTestSuite(log_path_prefix)
        self.assertIsInstance(summary, TestSummary)
        self.assertEqual(summary.passed, 2)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(summary.skipped, 0)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
"""
Testing utilities for PyTorch implementation.

This module extends the autotest framework with PyTorch-specific testing
capabilities, including tensor comparison, gradient checking, and
PyTorch-NumPy conversion for testing.
"""

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from eryx.autotest.testing import Testing
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping

class TorchTesting(Testing):
    """
    Extended testing framework for PyTorch implementations.
    
    This class extends the Testing class with PyTorch-specific utilities,
    including tensor comparison and gradient checking.
    """
    
    def __init__(self, logger: Logger, function_mapping: FunctionMapping, 
                rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize the PyTorch testing framework.
        
        Args:
            logger: Logger instance for test logging
            function_mapping: FunctionMapping instance for function lookup
            rtol: Relative tolerance for tensor comparison
            atol: Absolute tolerance for tensor comparison
        """
        super().__init__(logger, function_mapping)
        self.rtol = rtol
        self.atol = atol
    
    def testTorchCallable(self, log_path_prefix: str, torch_func: Callable) -> bool:
        """
        Test a PyTorch function against NumPy implementation.
        
        Args:
            log_path_prefix: Path prefix for log files
            torch_func: PyTorch function to test
            
        Returns:
            True if test passes, False otherwise
        """
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            for i in range(len(logs) // 2):
                args = logs[2 * i]['args']
                kwargs = logs[2 * i]['kwargs']
                expected_output = logs[2 * i + 1]['result']
                try:
                    deserialized_args = self._numpy_to_torch(self.logger.serializer.deserialize(args))
                    deserialized_kwargs = self._numpy_to_torch(self.logger.serializer.deserialize(kwargs))
                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
                    
                    actual_output = torch_func(*deserialized_args, **deserialized_kwargs)
                    
                    numpy_actual_output = self._torch_to_numpy(actual_output)
                    
                    if not self._compare_outputs(numpy_actual_output, deserialized_expected_output):
                        print(f"Test failed for {log_file}")
                        return False
                except Exception as e:
                    print(f"Error testing PyTorch function: {e}")
                    return False
        return True
    
    def _numpy_to_torch(self, obj: Any) -> Any:
        """
        Convert NumPy arrays to PyTorch tensors recursively.
        
        Args:
            obj: Input object potentially containing NumPy arrays
            
        Returns:
            Object with NumPy arrays converted to PyTorch tensors
        """
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj.copy())
        elif isinstance(obj, list):
            return [self._numpy_to_torch(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_torch(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._numpy_to_torch(v) for k, v in obj.items()}
        else:
            return obj
    
    def _torch_to_numpy(self, obj: Any) -> Any:
        """
        Convert PyTorch tensors to NumPy arrays recursively.
        
        Args:
            obj: Input object potentially containing PyTorch tensors
            
        Returns:
            Object with PyTorch tensors converted to NumPy arrays
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, list):
            return [self._torch_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._torch_to_numpy(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._torch_to_numpy(v) for k, v in obj.items()}
        else:
            return obj
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """
        Compare outputs with tolerance for numerical differences.
        
        Args:
            actual: Actual output
            expected: Expected output
            
        Returns:
            True if outputs match within tolerance, False otherwise
        """
        if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            if actual.shape != expected.shape:
                print(f"Shape mismatch: {actual.shape} vs {expected.shape}")
                return False
            
            nan_mask_actual = np.isnan(actual)
            nan_mask_expected = np.isnan(expected)
            if not np.array_equal(nan_mask_actual, nan_mask_expected):
                print("NaN pattern mismatch")
                return False
            
            non_nan_mask = ~nan_mask_actual
            if np.any(non_nan_mask):
                return np.allclose(actual[non_nan_mask], expected[non_nan_mask], 
                                 rtol=self.rtol, atol=self.atol)
            return True
        
        elif isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                print(f"Length mismatch: {len(actual)} vs {len(expected)}")
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, tuple) and isinstance(expected, tuple):
            if len(actual) != len(expected):
                print(f"Length mismatch: {len(actual)} vs {len(expected)}")
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                print(f"Key mismatch: {set(actual.keys())} vs {set(expected.keys())}")
                return False
            return all(self._compare_outputs(actual[k], expected[k]) for k in actual)
        
        else:
            return actual == expected
    
    def check_gradients(self, torch_func: Callable, inputs: List[torch.Tensor], 
                       eps: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
        """
        Check gradients of PyTorch function using finite differences.
        
        Args:
            torch_func: PyTorch function to check
            inputs: List of input tensors
            eps: Step size for finite differences
            
        Returns:
            Tuple containing:
                - True if gradients match, False otherwise
                - Dictionary with gradient statistics
        """
        # TODO: Clone inputs, set requires_grad=True, perform forward and backward passes,
        # calculate numerical gradients, and compare.
        raise NotImplementedError("check_gradients not implemented")
    
    def create_tensor_test_case(self, log_path_prefix: str, 
                              torch_func: Callable, numpy_func: Callable) -> Callable:
        """
        Create a test case for comparing PyTorch and NumPy implementations.
        
        Args:
            log_path_prefix: Path prefix for log files
            torch_func: PyTorch function to test
            numpy_func: NumPy function to compare against
            
        Returns:
            Test function that can be called directly
        """
        def test_case():
            # TODO: Generate test inputs, run both functions, and compare outputs.
            raise NotImplementedError("test_case not implemented")
        
        return test_case
