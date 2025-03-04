"""
Testing utilities for PyTorch implementation.

This module extends the autotest framework with PyTorch-specific testing
capabilities, including tensor comparison, gradient checking, and
PyTorch-NumPy conversion for testing.
"""

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .testing import Testing
from .logger import Logger
from .functionmapping import FunctionMapping

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
                    # Deserialize and convert to PyTorch
                    deserialized_args = self._numpy_to_torch(self.logger.serializer.deserialize(args))
                    deserialized_kwargs = self._numpy_to_torch(self.logger.serializer.deserialize(kwargs))
                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
                    
                    # Run PyTorch function
                    actual_output = torch_func(*deserialized_args, **deserialized_kwargs)
                    
                    # Convert PyTorch output to NumPy for comparison
                    numpy_actual_output = self._torch_to_numpy(actual_output)
                    
                    # Compare outputs
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
            # Compare arrays with tolerance
            if actual.shape != expected.shape:
                print(f"Shape mismatch: {actual.shape} vs {expected.shape}")
                return False
            
            # Handle NaN values
            nan_mask_actual = np.isnan(actual)
            nan_mask_expected = np.isnan(expected)
            if not np.array_equal(nan_mask_actual, nan_mask_expected):
                print("NaN pattern mismatch")
                return False
            
            # Compare non-NaN values
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
            # Direct comparison for other types
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
        # Clone inputs and set requires_grad=True
        grad_inputs = []
        for inp in inputs:
            grad_input = inp.clone().detach()
            grad_input.requires_grad_(True)
            grad_inputs.append(grad_input)
        
        # Compute forward pass
        output = torch_func(*grad_inputs)
        
        # If output is not a scalar, sum it to get a scalar
        if output.numel() > 1:
            output = output.sum()
        
        # Compute backward pass
        output.backward()
        
        # Get analytical gradients
        analytical_grads = [inp.grad.clone() for inp in grad_inputs]
        
        # Compute numerical gradients with finite differences
        numerical_grads = []
        for i, inp in enumerate(grad_inputs):
            numerical_grad = torch.zeros_like(inp)
            
            # Flatten the input for easier iteration
            flat_input = inp.flatten()
            flat_grad = numerical_grad.flatten()
            
            # Compute gradient for each element
            for j in range(flat_input.numel()):
                # Forward difference
                flat_input[j] += eps
                forward_output = torch_func(*grad_inputs).sum().item()
                
                # Backward difference
                flat_input[j] -= 2 * eps
                backward_output = torch_func(*grad_inputs).sum().item()
                
                # Central difference
                flat_grad[j] = (forward_output - backward_output) / (2 * eps)
                
                # Restore original value
                flat_input[j] += eps
            
            numerical_grads.append(numerical_grad)
        
        # Compare analytical and numerical gradients
        stats = {}
        all_match = True
        
        for i, (analytical, numerical) in enumerate(zip(analytical_grads, numerical_grads)):
            # Compute relative error
            abs_diff = torch.abs(analytical - numerical)
            abs_analytical = torch.abs(analytical)
            abs_numerical = torch.abs(numerical)
            
            # Avoid division by zero
            max_abs = torch.max(abs_analytical, abs_numerical)
            max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
            
            rel_error = abs_diff / max_abs
            
            # Compute statistics
            max_rel_error = rel_error.max().item()
            mean_rel_error = rel_error.mean().item()
            
            # Check if gradients match within tolerance
            match = (rel_error < self.rtol).all().item()
            all_match = all_match and match
            
            # Store statistics
            stats[f"input_{i}_max_rel_error"] = max_rel_error
            stats[f"input_{i}_mean_rel_error"] = mean_rel_error
            stats[f"input_{i}_match"] = match
        
        stats["all_match"] = all_match
        
        return all_match, stats
    
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
            # Find log files
            log_files = self.logger.searchLogDirectory(log_path_prefix)
            if not log_files:
                print(f"No log files found for {log_path_prefix}")
                return False
            
            # Process each log file
            for log_file in log_files:
                logs = self.logger.loadLog(log_file)
                for i in range(len(logs) // 2):
                    # Get inputs and expected output
                    args_bytes = logs[2 * i]['args']
                    kwargs_bytes = logs[2 * i]['kwargs']
                    
                    # Deserialize inputs
                    args = self.logger.serializer.deserialize(args_bytes)
                    kwargs = self.logger.serializer.deserialize(kwargs_bytes)
                    
                    # Run NumPy function to get expected output
                    expected_output = numpy_func(*args, **kwargs)
                    
                    # Convert inputs to PyTorch tensors
                    torch_args = self._numpy_to_torch(args)
                    torch_kwargs = self._numpy_to_torch(kwargs)
                    
                    # Run PyTorch function
                    torch_output = torch_func(*torch_args, **torch_kwargs)
                    
                    # Convert PyTorch output to NumPy
                    numpy_output = self._torch_to_numpy(torch_output)
                    
                    # Compare outputs
                    if not self._compare_outputs(numpy_output, expected_output):
                        print(f"Test failed for {log_file}")
                        return False
            
            return True
        
        return test_case
