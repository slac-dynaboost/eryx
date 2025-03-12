"""
Test module to verify the fix for the A_inv dictionary issue in StateBuilder.

This test specifically focuses on the issue where A_inv is serialized as a dictionary
instead of a numpy array, causing the error:
AttributeError: 'dict' object has no attribute 'size'
"""

import unittest
import torch
import numpy as np
import os
import sys
import io
import json
import pickle
from typing import Dict, Any

# Add parent directory to path to import eryx modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eryx.autotest.logger import Logger
from eryx.autotest.serializer import Serializer
from eryx.autotest.state_builder import StateBuilder
from eryx.models_torch import OnePhonon


class TestStateBuilderFix(unittest.TestCase):
    """Test case for verifying the StateBuilder fix for dictionary A_inv."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.logger = Logger()
        self.serializer = Serializer()
        
        # Create a mock state with A_inv as a dictionary
        self.mock_state = {
            'model': {
                'A_inv': {
                    'shape': (3, 3),
                    'dtype': 'float32',
                    '_array_type': 'numpy.ndarray'
                },
                'cell': np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
                'xyz': np.random.rand(2, 3, 3)
            },
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2]
        }
        
        # Create a temporary log file with the mock state
        os.makedirs('logs', exist_ok=True)
        self.log_path = 'logs/test_state_builder_fix.log'
        self._save_mock_state_log()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
    
    def _save_mock_state_log(self):
        """Save the mock state to a log file."""
        serialized_state = {}
        for key, value in self.mock_state.items():
            if key == 'model':
                # Serialize the model dictionary
                model_bytes = self.serializer.serialize(value)
                serialized_state[key] = model_bytes.hex()
            else:
                # Directly store other values
                serialized_state[key] = value
        
        with open(self.log_path, 'w') as f:
            json.dump(serialized_state, f)
    
    def test_build_with_dict_a_inv(self):
        """Test building a model with A_inv as a dictionary."""
        # Load the mock state
        state_data = self.logger.loadStateLog(self.log_path)
        
        # Create a StateBuilder
        builder = StateBuilder(device=self.device)
        
        # Build the model - this should not raise an exception
        try:
            model = builder.build(OnePhonon, state_data)
            
            # Verify A_inv was properly initialized
            self.assertTrue(hasattr(model, 'model'), "Model should have 'model' attribute")
            self.assertTrue(hasattr(model.model, 'A_inv'), "Model should have 'A_inv' attribute")
            self.assertIsInstance(model.model.A_inv, torch.Tensor, "A_inv should be a tensor")
            self.assertEqual(model.model.A_inv.shape, (3, 3), "A_inv should have shape (3, 3)")
            self.assertTrue(model.model.A_inv.requires_grad, "A_inv should require gradients")
            
            # Verify it's an identity matrix (our fallback)
            expected = torch.eye(3, device=self.device)
            self.assertTrue(torch.allclose(model.model.A_inv, expected), 
                           "A_inv should be initialized as identity matrix")
            
            print("✅ Test passed: StateBuilder correctly handles A_inv as dictionary")
        except Exception as e:
            self.fail(f"StateBuilder failed to build model with A_inv as dictionary: {e}")
    
    def test_deserialize_array_method(self):
        """Test the _deserialize_array method directly."""
        builder = StateBuilder(device=self.device)
        
        # Test with a dictionary containing shape and dtype
        dict_data = {
            'shape': (3, 3),
            'dtype': 'float32'
        }
        
        # Add debug print
        print(f"Testing _deserialize_array with: {dict_data}")
        result = builder._deserialize_array(dict_data)
        print(f"Result type: {type(result)}")
        if result is not None:
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            print(f"Result values:\n{result}")
            
        self.assertIsInstance(result, np.ndarray, "Result should be a numpy array")
        self.assertEqual(result.shape, (3, 3), "Result should have shape (3, 3)")
        self.assertEqual(result.dtype, np.float32, "Result should have dtype float32")
        
        # Verify it's an identity matrix for square matrices
        expected = np.eye(3, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected), 
                       "Result should be an identity matrix for square matrices")
        
        print("✅ Test passed: _deserialize_array correctly handles dictionary with shape and dtype")
        
        # Test with a dictionary containing actual array data
        test_array = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32)
        buffer = io.BytesIO()
        np.save(buffer, test_array)
        
        dict_data_with_values = {
            '_array_type': 'numpy.ndarray',
            '_array_data': buffer.getvalue(),
            '_array_dtype': str(test_array.dtype),
            '_array_shape': test_array.shape,
            '_array_values': test_array.tolist()
        }
        
        print(f"Testing _deserialize_array with array data")
        result_with_data = builder._deserialize_array(dict_data_with_values)
        print(f"Result type: {type(result_with_data)}")
        if result_with_data is not None:
            print(f"Result shape: {result_with_data.shape}, dtype: {result_with_data.dtype}")
            print(f"Result values:\n{result_with_data}")
            
        self.assertIsInstance(result_with_data, np.ndarray, "Result should be a numpy array")
        self.assertEqual(result_with_data.shape, test_array.shape, f"Result should have shape {test_array.shape}")
        self.assertEqual(result_with_data.dtype, test_array.dtype, f"Result should have dtype {test_array.dtype}")
        
        # Verify the values match
        self.assertTrue(np.allclose(result_with_data, test_array), 
                       "Result should match the original array values")
        
        print("✅ Test passed: _deserialize_array correctly handles dictionary with array data")
    
    def test_is_serialized_array_method(self):
        """Test the _is_serialized_array method directly."""
        builder = StateBuilder(device=self.device)
        
        # Test with a dictionary containing _array_type
        dict_data1 = {
            '_array_type': 'numpy.ndarray',
            '_array_data': b'dummy_data'
        }
        self.assertTrue(builder._is_serialized_array(dict_data1),
                       "Dictionary with _array_type should be identified as serialized array")
        
        # Test with a dictionary containing shape and dtype
        dict_data2 = {
            'shape': (3, 3),
            'dtype': 'float32'
        }
        self.assertTrue(builder._is_serialized_array(dict_data2),
                       "Dictionary with shape and dtype should be identified as serialized array")
        
        # Test with a regular dictionary
        dict_data3 = {
            'key1': 'value1',
            'key2': 'value2'
        }
        self.assertFalse(builder._is_serialized_array(dict_data3),
                        "Regular dictionary should not be identified as serialized array")
        
        print("✅ Test passed: _is_serialized_array correctly identifies serialized arrays")
    
    def test_tensor_creation(self):
        """Test tensor creation from state-restored model."""
        # Load the mock state
        state_data = self.logger.loadStateLog(self.log_path)
        
        # Build the model
        builder = StateBuilder(device=self.device)
        model = builder.build(OnePhonon, state_data)
        
        # Create a simple computation graph
        model._build_kvec_Brillouin()
        
        # Verify kvec was created
        self.assertTrue(hasattr(model, 'kvec'), "kvec should be created")
        self.assertTrue(model.kvec.requires_grad, "kvec should require gradients")
        
        # Note: We don't test gradient flow for state-restored instances
        # as per project requirements
        
        print("✅ Test passed: Tensor creation works correctly with state-restored model")
    
    def test_array_serialization_deserialization(self):
        """Test full serialization and deserialization of arrays."""
        from eryx.autotest.serializer import Serializer
        
        # Create a test array with non-identity values
        test_array = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32)
        
        # Create a serializer
        serializer = Serializer()
        
        # Serialize the array
        print(f"Serializing array with shape {test_array.shape} and dtype {test_array.dtype}")
        serialized_data = serializer.serialize(test_array)
        
        # Verify serialized data is bytes
        self.assertIsInstance(serialized_data, bytes, "Serialized data should be bytes")
        
        # Deserialize the array
        print(f"Deserializing array data")
        deserialized_array = serializer.deserialize(serialized_data)
        
        # Verify deserialized data is a numpy array with correct properties
        self.assertIsInstance(deserialized_array, np.ndarray, "Deserialized data should be a numpy array")
        self.assertEqual(deserialized_array.shape, test_array.shape, 
                        f"Deserialized array should have shape {test_array.shape}")
        self.assertEqual(deserialized_array.dtype, test_array.dtype, 
                        f"Deserialized array should have dtype {test_array.dtype}")
        
        # Verify the values match
        self.assertTrue(np.allclose(deserialized_array, test_array), 
                       "Deserialized array should match the original array values")
        
        print("✅ Test passed: Array serialization and deserialization works correctly")
        
        # Test with a model containing the array
        mock_model = {'A_inv': test_array}
        
        # Serialize the model
        print(f"Serializing model with A_inv array")
        serialized_model = serializer.serialize(mock_model)
        
        # Deserialize the model
        print(f"Deserializing model with A_inv array")
        deserialized_model = serializer.deserialize(serialized_model)
        
        # Verify the model contains the A_inv array with correct values
        self.assertIn('A_inv', deserialized_model, "Deserialized model should contain A_inv")
        self.assertIsInstance(deserialized_model['A_inv'], np.ndarray, 
                            "Deserialized A_inv should be a numpy array")
        self.assertTrue(np.allclose(deserialized_model['A_inv'], test_array), 
                       "Deserialized A_inv should match the original array values")
        
        print("✅ Test passed: Model with array serialization and deserialization works correctly")


    def test_state_capture_with_arrays(self):
        """Test state capture with arrays to ensure full serialization."""
        from eryx.autotest.state_capture import StateCapture
        from eryx.autotest.logger import Logger
        
        # Create a test object with arrays
        class TestObject:
            def __init__(self):
                self.A_inv = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32)
                self.simple_attr = "test"
                self.nested_obj = type('NestedObject', (), {
                    'nested_array': np.array([1.0, 2.0, 3.0], dtype=np.float64)
                })
        
        # Create a test object
        test_obj = TestObject()
        
        # Create a state capture instance
        state_capture = StateCapture(max_depth=3, include_private=False)
        
        # Capture the state
        print(f"Capturing state of test object")
        state = state_capture.capture_state(test_obj)
        
        # Verify state contains the expected attributes
        self.assertIn('A_inv', state, "State should contain A_inv")
        self.assertIn('simple_attr', state, "State should contain simple_attr")
        self.assertIn('nested_obj', state, "State should contain nested_obj")
        
        # Create a logger to serialize and deserialize the state
        logger = Logger()
        
        # Save the state to a temporary file
        temp_log_path = 'logs/test_state_capture.log'
        print(f"Saving state to {temp_log_path}")
        logger.saveStateLog(temp_log_path, state)
        
        # Load the state back
        print(f"Loading state from {temp_log_path}")
        loaded_state = logger.loadStateLog(temp_log_path)
        
        # Verify loaded state contains the expected attributes
        self.assertIn('A_inv', loaded_state, "Loaded state should contain A_inv")
        self.assertIn('simple_attr', loaded_state, "Loaded state should contain simple_attr")
        
        # Verify A_inv is a numpy array with correct values
        self.assertIsInstance(loaded_state['A_inv'], np.ndarray, 
                            "Loaded A_inv should be a numpy array")
        self.assertTrue(np.allclose(loaded_state['A_inv'], test_obj.A_inv), 
                       "Loaded A_inv should match the original array values")
        
        # Clean up the temporary file
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)
        
        print("✅ Test passed: State capture with arrays works correctly")

if __name__ == '__main__':
    unittest.main()
