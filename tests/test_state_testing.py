"""
Unit tests for state-based testing framework.

This module contains tests for the state-based testing capabilities,
including state capture, serialization, comparison, and initialization.
"""

import unittest
import numpy as np
import torch
import os
import tempfile
from typing import Dict, Any, List, Optional

from eryx.autotest.logger import Logger
from eryx.autotest.serializer import Serializer
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.functionmapping import FunctionMapping

# Mock classes for testing state capture
class SimpleNumPyClass:
    """Simple class with NumPy array attributes for testing."""
    
    def __init__(self):
        self.array1 = np.array([1.0, 2.0, 3.0])
        self.array2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.scalar = 42
        self.string = "test"
        self.list = [1, 2, 3]
        self.dict = {"a": 1, "b": 2}
    
    def modify_state(self):
        """Modify the object state for testing state changes."""
        self.array1 = np.array([4.0, 5.0, 6.0])
        self.scalar = 100
        self.list.append(4)
        self.dict["c"] = 3

class SimpleTorchClass:
    """Simple class with PyTorch tensor attributes for testing."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        self.tensor1 = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        self.tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        self.scalar = 42
        self.requires_grad = True
        
        # Enable gradients if specified
        if self.requires_grad:
            self.tensor1.requires_grad_(True)
            self.tensor2.requires_grad_(True)
    
    def modify_state(self):
        """Modify the object state for testing state changes."""
        self.tensor1 = torch.tensor([4.0, 5.0, 6.0], device=self.device)
        self.tensor1.requires_grad_(self.requires_grad)
        self.scalar = 100
    
    def compute_with_grad(self):
        """Perform computation that should flow gradients."""
        result = self.tensor1.sum() + self.tensor2.sum()
        result.backward()
        return result.item()

class ComplexStateClass:
    """Class with nested state for testing deep state capture."""
    
    def __init__(self):
        self.simple_numpy = SimpleNumPyClass()
        self.simple_torch = SimpleTorchClass()
        self.nested_dict = {
            "arrays": {
                "a": np.array([1.0, 2.0]),
                "b": np.array([3.0, 4.0])
            },
            "tensors": {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 4.0])
            }
        }
        self.nested_list = [
            np.array([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            [5.0, 6.0]
        ]
    
    def modify_nested_state(self):
        """Modify nested state for testing deep state changes."""
        self.simple_numpy.modify_state()
        self.simple_torch.modify_state()
        self.nested_dict["arrays"]["a"] = np.array([5.0, 6.0])
        self.nested_dict["tensors"]["a"] = torch.tensor([5.0, 6.0])
        self.nested_list[0] = np.array([7.0, 8.0])

class TestStateTesting(unittest.TestCase):
    """Test cases for state-based testing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger()
        self.serializer = Serializer()
        self.function_mapping = FunctionMapping()
        self.torch_testing = TorchTesting(self.logger, self.function_mapping)
        
        # Create temporary directory for test logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = self.temp_dir.name
        
        # Create test objects
        self.numpy_obj = SimpleNumPyClass()
        self.torch_obj = SimpleTorchClass()
        self.complex_obj = ComplexStateClass()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_state_capture(self):
        """Test capturing object state."""
        # Capture state of NumPy object
        numpy_state = self.logger.captureState(self.numpy_obj)
        
        # Verify state contains expected attributes
        self.assertIn('array1', numpy_state)
        self.assertIn('array2', numpy_state)
        self.assertIn('scalar', numpy_state)
        self.assertIn('string', numpy_state)
        self.assertIn('list', numpy_state)
        self.assertIn('dict', numpy_state)
        
        # Verify state values are serialized
        for key, value in numpy_state.items():
            self.assertIsInstance(value, bytes)
        
        # Capture state of PyTorch object
        torch_state = self.logger.captureState(self.torch_obj)
        
        # Verify state contains expected attributes
        self.assertIn('tensor1', torch_state)
        self.assertIn('tensor2', torch_state)
        self.assertIn('scalar', torch_state)
        self.assertIn('requires_grad', torch_state)
        
        # Test with include_private=True
        full_state = self.logger.captureState(self.numpy_obj, include_private=True)
        
        # Should include some private attributes if any exist
        private_attrs = [k for k in full_state.keys() if k.startswith('_')]
        # Note: SimpleNumPyClass might not have private attributes, so we don't assert length
        
        # Test with exclude_attrs
        filtered_state = self.logger.captureState(
            self.numpy_obj, exclude_attrs={'scalar', 'string'})
        
        self.assertNotIn('scalar', filtered_state)
        self.assertNotIn('string', filtered_state)
        self.assertIn('array1', filtered_state)
    
    def test_state_serialization(self):
        """Test serializing and deserializing state."""
        # Capture state
        state = self.logger.captureState(self.numpy_obj)
        
        # Save state to log file
        log_path = os.path.join(self.log_dir, 'test_state.log')
        self.logger.saveStateLog(log_path, state)
        
        # Verify log file exists
        self.assertTrue(os.path.exists(log_path))
        
        # Load state from log file
        loaded_state = self.logger.loadStateLog(log_path)
        
        # Verify loaded state matches original
        self.assertEqual(set(loaded_state.keys()), set(state.keys()))
        
        # Deserialize a specific attribute and verify value
        array1 = self.serializer.deserialize(state['array1'])
        loaded_array1 = loaded_state['array1']
        
        self.assertTrue(np.array_equal(array1, loaded_array1))
        
        # Test with complex object
        complex_state = self.logger.captureState(self.complex_obj)
        complex_log_path = os.path.join(self.log_dir, 'complex_state.log')
        self.logger.saveStateLog(complex_log_path, complex_state)
        
        loaded_complex_state = self.logger.loadStateLog(complex_log_path)
        self.assertEqual(set(loaded_complex_state.keys()), set(complex_state.keys()))
    
    def test_state_comparison(self):
        """Test comparing object states."""
        # Create two identical objects
        obj1 = SimpleNumPyClass()
        obj2 = SimpleNumPyClass()
        
        # Capture states
        state1 = obj1.__dict__
        state2 = obj2.__dict__
        
        # Compare states - should match
        self.assertTrue(self.torch_testing.compareStates(state1, state2))
        
        # Modify one object
        obj2.modify_state()
        state2 = obj2.__dict__
        
        # Compare states - should not match
        self.assertFalse(self.torch_testing.compareStates(state1, state2))
        
        # Test with custom tolerances
        obj3 = SimpleNumPyClass()
        obj3.array1 = np.array([1.001, 2.002, 3.003])  # Slightly different
        state3 = obj3.__dict__
        
        # Should fail with default tolerances
        self.assertFalse(self.torch_testing.compareStates(state1, state3))
        
        # Should pass with custom tolerances
        self.assertTrue(self.torch_testing.compareStates(
            state1, state3, attr_tolerances={'array1': {'rtol': 0.01, 'atol': 0.01}}))
        
        # Test with PyTorch tensors
        torch_obj1 = SimpleTorchClass()
        torch_obj2 = SimpleTorchClass()
        
        torch_state1 = torch_obj1.__dict__
        torch_state2 = torch_obj2.__dict__
        
        # Should match
        self.assertTrue(self.torch_testing.compareStates(torch_state1, torch_state2))
        
        # Modify tensor slightly
        torch_obj2.tensor1 = torch.tensor([1.001, 2.002, 3.003])
        torch_state2 = torch_obj2.__dict__
        
        # Should fail with default tolerances
        self.assertFalse(self.torch_testing.compareStates(torch_state1, torch_state2))
        
        # Should pass with custom tolerances
        self.assertTrue(self.torch_testing.compareStates(
            torch_state1, torch_state2, 
            attr_tolerances={'tensor1': {'rtol': 0.01, 'atol': 0.01}}))
    
    def test_initialization_from_state(self):
        """Test initializing objects from state."""
        # Capture state of NumPy object
        numpy_state = self.logger.captureState(self.numpy_obj)
        
        # Initialize new object from state
        new_numpy_obj = self.torch_testing.initializeFromState(SimpleNumPyClass, numpy_state)
        
        # Verify attributes match
        self.assertTrue(np.array_equal(new_numpy_obj.array1, self.numpy_obj.array1))
        self.assertTrue(np.array_equal(new_numpy_obj.array2, self.numpy_obj.array2))
        self.assertEqual(new_numpy_obj.scalar, self.numpy_obj.scalar)
        self.assertEqual(new_numpy_obj.string, self.numpy_obj.string)
        self.assertEqual(new_numpy_obj.list, self.numpy_obj.list)
        self.assertEqual(new_numpy_obj.dict, self.numpy_obj.dict)
        
        # Capture state of PyTorch object
        torch_state = self.logger.captureState(self.torch_obj)
        
        # Initialize new object from state
        new_torch_obj = self.torch_testing.initializeFromState(SimpleTorchClass, torch_state)
        
        # Verify attributes match
        self.assertTrue(torch.allclose(new_torch_obj.tensor1, self.torch_obj.tensor1))
        self.assertTrue(torch.allclose(new_torch_obj.tensor2, self.torch_obj.tensor2))
        self.assertEqual(new_torch_obj.scalar, self.torch_obj.scalar)
        
        # Device placement tests removed - tensors are now restored on their original device
        
        # Test with complex object
        complex_state = self.logger.captureState(self.complex_obj)
        new_complex_obj = self.torch_testing.initializeFromState(ComplexStateClass, complex_state)
        
        # Verify nested attributes
        self.assertTrue(np.array_equal(
            new_complex_obj.simple_numpy.array1, self.complex_obj.simple_numpy.array1))
        self.assertTrue(torch.allclose(
            new_complex_obj.simple_torch.tensor1, self.complex_obj.simple_torch.tensor1))
    
    def test_gradient_preservation(self):
        """Test gradient flow in initialized objects."""
        # Create PyTorch object with gradients
        torch_obj = SimpleTorchClass()
        torch_obj.tensor1.requires_grad_(True)
        torch_obj.tensor2.requires_grad_(True)
        
        # Capture state
        torch_state = self.logger.captureState(torch_obj)
        
        # Initialize new object from state
        new_torch_obj = self.torch_testing.initializeFromState(SimpleTorchClass, torch_state)
        
        # Verify gradients are enabled
        self.assertTrue(new_torch_obj.tensor1.requires_grad)
        self.assertTrue(new_torch_obj.tensor2.requires_grad)
        
        # Perform computation that should flow gradients
        result = new_torch_obj.compute_with_grad()
        
        # Verify gradients were computed
        self.assertIsNotNone(new_torch_obj.tensor1.grad)
        self.assertIsNotNone(new_torch_obj.tensor2.grad)
        
        # Check gradient values
        self.assertTrue(torch.all(new_torch_obj.tensor1.grad == 1.0))
        self.assertTrue(torch.all(new_torch_obj.tensor2.grad == 1.0))
        
        # Test check_state_gradients method
        grad_status = self.torch_testing.check_state_gradients(new_torch_obj)
        self.assertTrue(grad_status['tensor1'])
        self.assertTrue(grad_status['tensor2'])
    
    def test_end_to_end_state_testing(self):
        """Test complete state-based testing workflow."""
        # Create test object
        obj = SimpleTorchClass()
        
        # Capture before state
        before_state = self.logger.captureState(obj)
        before_log_path = os.path.join(self.log_dir, 'before_state.log')
        self.logger.saveStateLog(before_log_path, before_state)
        
        # Modify object state
        obj.modify_state()
        
        # Capture after state
        after_state = self.logger.captureState(obj)
        after_log_path = os.path.join(self.log_dir, 'after_state.log')
        self.logger.saveStateLog(after_log_path, after_state)
        
        # Create new object from before state
        new_obj = self.torch_testing.initializeFromState(SimpleTorchClass, before_state)
        
        # Verify initial state matches
        self.assertTrue(torch.allclose(new_obj.tensor1, torch.tensor([1.0, 2.0, 3.0])))
        self.assertEqual(new_obj.scalar, 42)
        
        # Call method that modifies state
        new_obj.modify_state()
        
        # Verify state changed as expected
        self.assertTrue(torch.allclose(new_obj.tensor1, torch.tensor([4.0, 5.0, 6.0])))
        self.assertEqual(new_obj.scalar, 100)
        
        # Compare with expected after state
        expected_after_state = self.logger.loadStateLog(after_log_path)
        self.assertTrue(self.torch_testing.compareStates(expected_after_state, new_obj.__dict__))
        
        # Test testTorchCallableWithState method
        # First, set up log paths to match expected pattern
        log_prefix = os.path.join(self.log_dir, 'test_module.SimpleTorchClass')
        before_method_log = f"{log_prefix}._state_before_modify_state.log"
        after_method_log = f"{log_prefix}._state_after_modify_state.log"
        
        # Save states with correct naming
        self.logger.saveStateLog(before_method_log, before_state)
        self.logger.saveStateLog(after_method_log, after_state)
        
        # Test the method
        result = self.torch_testing.testTorchCallableWithState(
            log_prefix, SimpleTorchClass, 'modify_state')
        
        # Should pass
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
