"""
Tests for the ObjectSerializer class.
"""

import unittest
import json
import io
import os
import sys
import tempfile
from typing import Dict, List, Any, Optional

import numpy as np

from eryx.serialization import ObjectSerializer, SerializationError, DeserializationError


class TestObjectSerializer(unittest.TestCase):
    """Test cases for ObjectSerializer."""
    
    def setUp(self):
        """Set up test environment."""
        self.serializer = ObjectSerializer()
    
    def test_basic_types(self):
        """Test serialization/deserialization of basic types."""
        test_cases = [
            42,
            3.14159,
            "hello world",
            True,
            False,
            None
        ]
        
        for value in test_cases:
            serialized = self.serializer.serialize(value)
            deserialized = self.serializer.deserialize(serialized)
            self.assertEqual(value, deserialized)
            
            # Also test JSON round-trip
            json_str = self.serializer.dumps(value)
            from_json = self.serializer.loads(json_str)
            self.assertEqual(value, from_json)
    
    def test_collection_types(self):
        """Test serialization/deserialization of collections."""
        test_cases = [
            [1, 2, 3, 4, 5],
            (1, "two", 3.0),
            {"key1": "value1", "key2": 42},
            {1, 2, 3, 4, 5},
            [1, [2, 3], {"key": "value"}],
            {"nested": {"list": [1, 2, 3]}}
        ]
        
        for value in test_cases:
            serialized = self.serializer.serialize(value)
            deserialized = self.serializer.deserialize(serialized)
            self.assertEqual(value, deserialized)
    
    def test_numpy_arrays(self):
        """Test NumPy array serialization/deserialization."""
        try:
            # Test various array types
            arrays = [
                np.array([1, 2, 3, 4, 5]),
                np.array([[1, 2], [3, 4]]),
                np.array([1.1, 2.2, 3.3]),
                np.zeros((3, 3)),
                np.ones((2, 2, 2)),
                np.array([1+2j, 3+4j])
            ]
            
            for array in arrays:
                serialized = self.serializer.serialize(array)
                deserialized = self.serializer.deserialize(serialized)
                np.testing.assert_array_equal(array, deserialized)
                
        except ImportError:
            self.skipTest("NumPy not available")
    
    def test_gemmi_structure(self):
        """Test Gemmi Structure serialization/deserialization."""
        try:
            import gemmi
            
            # Create a simple Structure with basic components
            structure = gemmi.Structure()
            structure.name = "test_structure"
            structure.cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
            structure.spacegroup_hm = "P 1"
            
            model = gemmi.Model("1")
            chain = gemmi.Chain("A")
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.seqid = gemmi.SeqId(1, ' ')
            
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.pos = gemmi.Position(1, 2, 3)
            atom.element = gemmi.Element("C")
            atom.b_iso = 20.0
            atom.occ = 1.0
            
            residue.add_atom(atom)
            chain.add_residue(residue)
            model.add_chain(chain)
            structure.add_model(model)
            
            # Test serialization/deserialization
            serialized = self.serializer.serialize(structure)
            self.assertEqual(serialized["__type__"], "gemmi.Structure")
            self.assertEqual(serialized["name"], "test_structure")
            
            # Test deserialization if GemmiSerializer is available
            try:
                from eryx.autotest.gemmi_serializer import GemmiSerializer
                deserialized = self.serializer.deserialize(serialized)
                self.assertEqual(deserialized.name, "test_structure")
                self.assertEqual(len(deserialized), 1)  # One model
                self.assertEqual(len(deserialized[0]), 1)  # One chain
            except ImportError:
                print("GemmiSerializer not available, skipping deserialization test")
                
        except ImportError:
            self.skipTest("Gemmi not available")
    
    def test_custom_objects(self):
        """Test object serialization/deserialization."""
        # Define a simple custom class
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age
                
            def __eq__(self, other):
                if not isinstance(other, Person):
                    return False
                return self.name == other.name and self.age == other.age
        
        # Test simple object
        person = Person("Alice", 30)
        serialized = self.serializer.serialize(person)
        deserialized = self.serializer.deserialize(serialized)
        
        self.assertEqual(deserialized.name, "Alice")
        self.assertEqual(deserialized.age, 30)
        
        # Test nested objects
        class Department:
            def __init__(self, name, manager):
                self.name = name
                self.manager = manager
                self.employees = []
        
        dept = Department("Engineering", Person("Bob", 45))
        dept.employees = [Person("Charlie", 35), Person("Dave", 28)]
        
        serialized = self.serializer.serialize(dept)
        deserialized = self.serializer.deserialize(serialized)
        
        self.assertEqual(deserialized.name, "Engineering")
        self.assertEqual(deserialized.manager.name, "Bob")
        self.assertEqual(deserialized.manager.age, 45)
        self.assertEqual(len(deserialized.employees), 2)
        self.assertEqual(deserialized.employees[0].name, "Charlie")
        self.assertEqual(deserialized.employees[1].name, "Dave")
    
    def test_error_handling(self):
        """Test behavior with problematic inputs."""
        # Test deserializing invalid data
        with self.assertRaises(DeserializationError):
            self.serializer.deserialize({"invalid": "data"})
        
        # Test deserializing non-dict
        with self.assertRaises(DeserializationError):
            self.serializer.deserialize("not a dict")
        
        # Test serializing unserializable object
        class Unserializable:
            def __init__(self):
                self.circular_ref = self
        
        obj = Unserializable()
        with self.assertRaises(SerializationError):
            self.serializer.serialize(obj)
    
    def test_file_io(self):
        """Test file I/O operations."""
        data = {
            "name": "Test Data",
            "values": [1, 2, 3, 4, 5],
            "nested": {"key": "value"}
        }
        
        # Test dump/load with file objects
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            filename = f.name
            self.serializer.dump(data, f)
        
        try:
            with open(filename, 'r') as f:
                loaded = self.serializer.load(f)
            
            self.assertEqual(data, loaded)
        finally:
            os.unlink(filename)
    
    def test_custom_handler(self):
        """Test registering and using custom type handlers."""
        # Define a custom class
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
            def __eq__(self, other):
                if not isinstance(other, Point):
                    return False
                return self.x == other.x and self.y == other.y
        
        # Define custom handlers
        def serialize_point(point):
            return {
                "__type__": "custom.Point",
                "x": point.x,
                "y": point.y
            }
        
        def deserialize_point(data):
            return Point(data["x"], data["y"])
        
        # Register handlers
        self.serializer.register_handler(Point, serialize_point, deserialize_point)
        
        # Test serialization/deserialization
        point = Point(10, 20)
        serialized = self.serializer.serialize(point)
        deserialized = self.serializer.deserialize(serialized)
        
        self.assertEqual(point, deserialized)
        self.assertEqual(deserialized.x, 10)
        self.assertEqual(deserialized.y, 20)

    def test_numpy_array_format(self):
        """Test the specific binary format used for NumPy array serialization."""
        try:
            # Create test arrays of different shapes and types
            array1 = np.array([1, 2, 3, 4, 5])
            array2 = np.array([[1.1, 2.2], [3.3, 4.4]])
            
            # Test array1 serialization format
            serialized = self.serializer.serialize(array1)
            
            # Check required fields
            self.assertEqual(serialized["__type__"], "numpy.ndarray")
            self.assertEqual(serialized["__shape__"], (5,))
            self.assertEqual(serialized["__dtype__"], str(array1.dtype))
            self.assertTrue("__binary__" in serialized)
            
            # Verify binary data is valid base64
            import base64
            binary_data = serialized["__binary__"]
            try:
                decoded = base64.b64decode(binary_data)
                self.assertTrue(len(decoded) > 0)
            except Exception as e:
                self.fail(f"Failed to decode binary data: {e}")
            
            # Test that deserialization works with the binary format
            deserialized = self.serializer.deserialize(serialized)
            np.testing.assert_array_equal(array1, deserialized)
            
            # Test 2D array
            serialized2 = self.serializer.serialize(array2)
            deserialized2 = self.serializer.deserialize(serialized2)
            np.testing.assert_array_equal(array2, deserialized2)
            
            # Test fallback mechanism
            fallback_data = {
                "__type__": "numpy.ndarray",
                "__shape__": (3, 3),
                "__dtype__": "float64"
                # No __binary__ field
            }
            fallback_array = self.serializer.deserialize(fallback_data)
            self.assertEqual(fallback_array.shape, (3, 3))
            self.assertEqual(fallback_array.dtype, np.dtype('float64'))
            
        except ImportError:
            self.skipTest("NumPy not available")

if __name__ == '__main__':
    unittest.main()
