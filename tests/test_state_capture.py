"""
Tests for the state capture functionality.
"""
import unittest
import os
import tempfile
import numpy as np
from typing import Dict, Any, List

from eryx.autotest.state_capture import StateCapture
from eryx.autotest.logger import Logger
from eryx.autotest.debug import debug

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TestObject:
    """Simple test object with known attributes for testing state capture."""
    
    def __init__(self):
        # Basic attributes
        self.int_attr = 42
        self.float_attr = 3.14
        self.string_attr = "test"
        self.list_attr = [1, 2, 3]
        self.dict_attr = {"key": "value"}
        
        # Nested attribute
        self.nested_attr = {"a": [1, 2, 3], "b": {"c": "nested"}}
        
        # Private attribute
        self._private_attr = "private"
        
        # Array attribute
        self.array_attr = np.array([1.0, 2.0, 3.0])
        
        # Tensor attribute if torch is available
        if TORCH_AVAILABLE:
            self.tensor_attr = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
    def method(self) -> int:
        """Method that should be skipped by state capture."""
        return self.int_attr
        
    @debug
    def modify_state(self, new_value: int) -> None:
        """Method decorated with debug to test state capture."""
        self.int_attr = new_value
        self.list_attr.append(new_value)

class TestStateCapture(unittest.TestCase):
    """Tests for StateCapture class and related functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_obj = TestObject()
        self.state_capturer = StateCapture()
        self.logger = Logger()
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_attribute_exclusion(self):
        """Test excluding specific attributes."""
        capturer = StateCapture(exclude_attrs=['int_.*', 'nested_.*'])
        state = capturer.capture_state(self.test_obj)
        
        # Verify excluded attributes were not captured
        self.assertNotIn('int_attr', state)
        self.assertNotIn('nested_attr', state)
        
        # Verify other attributes were still captured
        self.assertIn('float_attr', state)
        self.assertIn('string_attr', state)
    
    @unittest.skipIf(os.environ.get("DEBUG_MODE") != "1", "Debug mode not enabled")
    def test_debug_decorator_state_capture(self):
        """Test state capture via the debug decorator."""
        # Call method with debug decorator that modifies state
        self.test_obj.modify_state(100)
        
        # Check for state log files
        module_path = self.test_obj.__class__.__module__
        if not module_path:
            module_path = "__main__"
            
        class_name = self.test_obj.__class__.__name__
        method_name = "modify_state"
        
        before_path = f"logs/{module_path}.{class_name}._state_before_{method_name}.log"
        after_path = f"logs/{module_path}.{class_name}._state_after_{method_name}.log"
        
        # Verify log files exist
        self.assertTrue(os.path.exists(before_path), f"Before state log not found: {before_path}")
        self.assertTrue(os.path.exists(after_path), f"After state log not found: {after_path}")
            
        # Load states and verify they show the change
        before_state = self.logger.loadStateLog(before_path)
        after_state = self.logger.loadStateLog(after_path)
        
        self.assertEqual(before_state.get('int_attr'), 42)
        self.assertEqual(after_state.get('int_attr'), 100)
        self.assertEqual(before_state.get('list_attr'), [1, 2, 3])
if __name__ == '__main__':
    unittest.main()
