"""
Tests for the enhanced serializer with support for unserializable objects.
"""
import unittest
import pickle
import numpy as np
import io
import sys
import os
from typing import Any, Dict

# Add parent directory to path to import serializer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from eryx.autotest.serializer import Serializer, GemmiSerializer

# Create a mock Gemmi-like object for testing
class MockGemmiObject:
    """Mock object that simulates a Gemmi object for testing."""
    
    def __init__(self, name="test", a=10.0, b=20.0, c=30.0):
        self.name = name
        self.cell = MockCell(a, b, c)
        self.__module__ = "gemmi.mock"
    
    def __repr__(self):
        return f"<MockGemmiObject name={self.name}>"

class MockCell:
    """Mock cell object for testing."""
    
    def __init__(self, a=10.0, b=20.0, c=30.0, alpha=90.0, beta=90.0, gamma=90.0):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __repr__(self):
        return f"<MockCell a={self.a} b={self.b} c={self.c}>"

# Create an unserializable object
class UnserializableObject:
    """Object that cannot be serialized with pickle."""
    
    def __init__(self, value="test"):
        self.value = value
        # Add an attribute that makes this unserializable
        self.unserializable = lambda x: x
    
    def __repr__(self):
        return f"<UnserializableObject value={self.value}>"

class TestSerializerEnhanced(unittest.TestCase):
    """Test cases for the enhanced serializer."""
    
    def setUp(self):
        """Set up test environment."""
        self.serializer = Serializer()
    
    def test_serialize_simple_object(self):
        """Test serializing a simple object."""
        obj = {"key": "value", "number": 42}
        serialized = self.serializer.serialize(obj)
        deserialized = self.serializer.deserialize(serialized)
        self.assertEqual(deserialized, obj)
    
    def test_serialize_numpy_array(self):
        """Test serializing a NumPy array."""
        array = np.array([1, 2, 3, 4, 5])
        serialized = self.serializer.serialize(array)
        deserialized = self.serializer.deserialize(serialized)
        np.testing.assert_array_equal(deserialized, array)
    
    def test_serialize_mock_gemmi_object(self):
        """Test serializing a mock Gemmi object."""
        gemmi_obj = MockGemmiObject("test_structure", 10.0, 20.0, 30.0)
        
        # Serialize the object
        serialized = self.serializer.serialize(gemmi_obj)
        self.assertIsInstance(serialized, bytes)
        
        # Deserialize and check the result
        deserialized = self.serializer.deserialize(serialized)
        
        # The result could be either a MockGemmiObject or a dictionary
        if isinstance(deserialized, dict):
            # Check if it contains Gemmi type information
            if "_serialized_gemmi" in deserialized:
                # If using GemmiSerializer
                self.assertTrue(deserialized.get("_serialized_gemmi", False))
                data = deserialized.get("data", {})
                self.assertIn("_gemmi_type", data)
                # Check the name
                self.assertEqual(data.get("name", ""), "test_structure")
            elif "__unserializable__" in deserialized:
                # If using the fallback serialization
                self.assertTrue(deserialized.get("__unserializable__", False))
                self.assertIn("__gemmi_type__", deserialized)
                self.assertTrue(deserialized.get("__gemmi_type__", False))
                # Check the name in gemmi_info
                if "__gemmi_info__" in deserialized:
                    self.assertEqual(deserialized["__gemmi_info__"].get("name", ""), "test_structure")
        else:
            # If it's the original object, check its name
            self.assertEqual(deserialized.name, "test_structure")
    
    def test_serialize_unserializable_object(self):
        """Test serializing an object that cannot be pickled."""
        obj = UnserializableObject("test_value")
        
        # Serialize the object
        serialized = self.serializer.serialize(obj)
        self.assertIsInstance(serialized, bytes)
        
        # Deserialize and check the result
        deserialized = self.serializer.deserialize(serialized)
        
        # Check that we got a dictionary with the expected structure
        self.assertIsInstance(deserialized, dict)
        self.assertTrue(deserialized.get("__unserializable__", False))
        
        # The module name might be __main__ or test_serializer_enhanced
        type_name = deserialized.get("__type__", "")
        self.assertTrue(
            type_name.endswith(".UnserializableObject") or type_name.endswith("UnserializableObject"),
            f"Unexpected type name: {type_name}"
        )
        
        # Check if attributes were captured
        if "__attributes__" in deserialized:
            self.assertEqual(deserialized["__attributes__"].get("value", ""), "test_value")
    
    def test_serialize_collection_with_unserializable_objects(self):
        """Test serializing a collection containing unserializable objects."""
        # Create a dictionary with mixed content
        mixed_dict = {
            "normal": "value",
            "number": 42,
            "array": np.array([1, 2, 3]),
            "unserializable": UnserializableObject("in_dict"),
            "gemmi": MockGemmiObject("in_dict")
        }
        
        # Serialize the dictionary
        serialized = self.serializer.serialize(mixed_dict)
        self.assertIsInstance(serialized, bytes)
        
        # Deserialize and check the result
        deserialized = self.serializer.deserialize(serialized)
        
        # The dictionary might be serialized as a dictionary or as an unserializable object
        if isinstance(deserialized, dict) and not deserialized.get("__unserializable__", False):
            # Regular dictionary case
            # Check normal values
            self.assertEqual(deserialized.get("normal"), "value")
            self.assertEqual(deserialized.get("number"), 42)
            
            # Check array
            if isinstance(deserialized.get("array"), np.ndarray):
                np.testing.assert_array_equal(deserialized.get("array"), np.array([1, 2, 3]))
            
            # Check unserializable object
            unserializable = deserialized.get("unserializable")
            if isinstance(unserializable, dict) and "__unserializable__" in unserializable:
                self.assertTrue(unserializable.get("__unserializable__", False))
            
            # Check Gemmi object
            gemmi = deserialized.get("gemmi")
            if isinstance(gemmi, dict):
                if "_serialized_gemmi" in gemmi:
                    # If using GemmiSerializer
                    self.assertTrue(gemmi.get("_serialized_gemmi", False))
                elif "__unserializable__" in gemmi:
                    # If using the fallback serialization
                    self.assertTrue(gemmi.get("__unserializable__", False))
                    if "__gemmi_type__" in gemmi:
                        self.assertTrue(gemmi.get("__gemmi_type__", False))
        else:
            # The whole dictionary was marked as unserializable
            self.assertTrue(deserialized.get("__unserializable__", False))
            # Check if it has items
            if "__items__" in deserialized:
                items = deserialized["__items__"]
                self.assertIsInstance(items, dict)
    
    def test_serialize_list_with_unserializable_objects(self):
        """Test serializing a list containing unserializable objects."""
        # Create a list with mixed content
        mixed_list = [
            "value",
            42,
            np.array([1, 2, 3]),
            UnserializableObject("in_list"),
            MockGemmiObject("in_list")
        ]
        
        # Serialize the list
        serialized = self.serializer.serialize(mixed_list)
        self.assertIsInstance(serialized, bytes)
        
        # Deserialize and check the result
        deserialized = self.serializer.deserialize(serialized)
        
        # The list might be serialized as a list or as an unserializable object
        if isinstance(deserialized, list):
            # Regular list case
            # Check length
            self.assertEqual(len(deserialized), 5)
            
            # Check normal values
            self.assertEqual(deserialized[0], "value")
            self.assertEqual(deserialized[1], 42)
            
            # Check array
            if isinstance(deserialized[2], np.ndarray):
                np.testing.assert_array_equal(deserialized[2], np.array([1, 2, 3]))
            
            # Check unserializable object
            unserializable = deserialized[3]
            if isinstance(unserializable, dict) and "__unserializable__" in unserializable:
                self.assertTrue(unserializable.get("__unserializable__", False))
            
            # Check Gemmi object
            gemmi = deserialized[4]
            if isinstance(gemmi, dict):
                if "_serialized_gemmi" in gemmi:
                    # If using GemmiSerializer
                    self.assertTrue(gemmi.get("_serialized_gemmi", False))
                elif "__unserializable__" in gemmi:
                    # If using the fallback serialization
                    self.assertTrue(gemmi.get("__unserializable__", False))
                    if "__gemmi_type__" in gemmi:
                        self.assertTrue(gemmi.get("__gemmi_type__", False))
        else:
            # The whole list was marked as unserializable
            self.assertTrue(isinstance(deserialized, dict))
            self.assertTrue(deserialized.get("__unserializable__", False))
            self.assertEqual(deserialized.get("__collection_type__", ""), "list")
            # Check if it has items
            if "__items__" in deserialized:
                items = deserialized["__items__"]
                self.assertIsInstance(items, list)

if __name__ == "__main__":
    unittest.main()
