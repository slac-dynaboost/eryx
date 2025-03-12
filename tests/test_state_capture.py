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
    
    def test_capture_basic_attributes(self):
        """Test capturing basic attributes."""
        state = self.state_capturer.capture_state(self.test_obj)
        
        # Verify basic attributes were captured
        self.assertEqual(self.logger.serializer.deserialize(state.get('int_attr')), 42)
        self.assertAlmostEqual(self.logger.serializer.deserialize(state.get('float_attr')), 3.14)
        self.assertEqual(self.logger.serializer.deserialize(state.get('string_attr')), "test")
        self.assertEqual(self.logger.serializer.deserialize(state.get('list_attr')), [1, 2, 3])
        self.assertEqual(self.logger.serializer.deserialize(state.get('dict_attr')), {"key": "value"})
        
        # Verify nested attributes were captured
        nested = self.logger.serializer.deserialize(state.get('nested_attr'))
        self.assertEqual(nested, {"a": [1, 2, 3], "b": {"c": "nested"}})
        
        # Verify methods were not captured
        self.assertNotIn('method', state)
        
        # Verify private attributes were not captured by default
        self.assertNotIn('_private_attr', state)
    
    def test_private_attribute_inclusion(self):
        """Test including private attributes."""
        capturer = StateCapture(include_private=True)
        state = capturer.capture_state(self.test_obj)
        
        # Verify private attribute was captured
        self.assertEqual(self.logger.serializer.deserialize(state.get('_private_attr')), "private")
    
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
    
    def test_max_depth_limitation(self):
        """Test max depth limitation."""
        # Create a simple nested dictionary structure instead of classes
        # (to avoid pickling issues with local classes)
        nested_obj = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 42
                    }
                }
            }
        }
        
        # Test with max_depth=1
        capturer = StateCapture(max_depth=1)
        state = capturer.capture_state(nested_obj)
        
        # Verify first level was captured
        self.assertIn('level1', state)
        
        # Load the level1 state
        level1_state = self.logger.serializer.deserialize(state['level1'])
        
        # Verify it has the max depth marker
        self.assertIn('__max_depth_reached__', level1_state)
        self.assertTrue(level1_state['__max_depth_reached__'])
    
    def test_serialization_and_deserialization(self):
        """Test serialization and deserialization of state."""
        # Capture state
        state = self.state_capturer.capture_state(self.test_obj)
        
        # Save to temp file
        temp_file = os.path.join(self.temp_dir.name, "test_state.log")
        self.logger.saveStateLog(temp_file, state)
        
        # Load from temp file
        loaded_state = self.logger.loadStateLog(temp_file)
        
        # Verify state was preserved
        self.assertEqual(loaded_state.get('int_attr'), 42)
        self.assertEqual(loaded_state.get('list_attr'), [1, 2, 3])
        
        # Check numpy array was properly serialized and deserialized
        if 'array_attr' in loaded_state:
            np.testing.assert_array_equal(
                loaded_state.get('array_attr'), 
                np.array([1.0, 2.0, 3.0])
            )
        
        # Check tensor if available
        if TORCH_AVAILABLE and 'tensor_attr' in loaded_state:
            tensor_data = loaded_state.get('tensor_attr')
            if isinstance(tensor_data, torch.Tensor):
                self.assertTrue(torch.allclose(
                    tensor_data,
                    torch.tensor([1.0, 2.0, 3.0])
                ))
            else:
                # Might be converted to numpy
                np.testing.assert_array_almost_equal(
                    tensor_data,
                    np.array([1.0, 2.0, 3.0])
                )
    
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
        self.assertEqual(after_state.get('list_attr'), [1, 2, 3, 100])

if __name__ == '__main__':
    unittest.main()
