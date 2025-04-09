#!/usr/bin/env python3
"""
Test module for GaussianNetworkModel serialization and deserialization.

This script tests the serialization and deserialization of GaussianNetworkModel
objects to ensure they maintain the correct structure and attributes.
"""

import os
import sys
import json
import torch
import numpy as np
import unittest
from typing import Dict, Any

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eryx.pdb import GaussianNetworkModel as NumpyGNM
from eryx.pdb_torch import GaussianNetworkModel as TorchGNM
from eryx.autotest.logger import Logger
from eryx.autotest.serializer import Serializer
from eryx.autotest.state_builder import StateBuilder

class TestGNMSerialization(unittest.TestCase):
    """Test case for GaussianNetworkModel serialization and deserialization."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = Logger()
        self.serializer = Serializer()
        self.test_file = "test_gnm_serialization.log"
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameters for GNM
        self.enm_cutoff = 4.0
        self.gamma_intra = 1.0
        self.gamma_inter = 1.0
        
        # Clean up any existing test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_numpy_gnm_serialization(self):
        """Test serialization and deserialization of NumPy GNM."""
        # Create a NumPy GNM instance
        gnm = NumpyGNM(self.pdb_path, self.enm_cutoff, self.gamma_intra, self.gamma_inter)
        
        # Serialize the GNM
        state = self._capture_state(gnm)
        
        # Create a custom encoder to handle non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return {'__type__': obj.__class__.__name__, 'attributes': str(obj)}
                return str(obj)
        
        # Save to file
        with open(self.test_file, 'w') as f:
            json.dump(state, f, indent=2, cls=CustomEncoder)
        
        # Load from file
        with open(self.test_file, 'r') as f:
            loaded_state = json.load(f)
        
        # Verify structure
        self._verify_gnm_state(loaded_state)
        
        # Print state structure
        self._print_state_structure(loaded_state)
    
    def _capture_state(self, obj: Any) -> Dict[str, Any]:
        """Capture the state of an object."""
        state = {}
        for attr_name in dir(obj):
            if attr_name.startswith('_') or callable(getattr(obj, attr_name)):
                continue
            try:
                attr_value = getattr(obj, attr_name)
                # Handle non-serializable objects
                if attr_name == 'crystal':
                    # Store only essential properties from crystal
                    state[attr_name] = {
                        'n_cell': getattr(attr_value, 'n_cell', 0),
                        # Store function indicators that will be replaced with actual functions
                        'id_to_hkl': '__function__',
                        'get_unitcell_origin': '__function__'
                    }
                elif isinstance(attr_value, np.ndarray):
                    # Convert numpy arrays to lists for JSON serialization
                    state[attr_name] = {
                        'shape': attr_value.shape,
                        'dtype': str(attr_value.dtype),
                        'data': attr_value.tolist() if attr_value.size < 1000 else 'large_array'
                    }
                else:
                    state[attr_name] = attr_value
            except Exception as e:
                print(f"Warning: Could not capture {attr_name}: {e}")
        return state
    
    def test_state_builder_gnm_handler(self):
        """Test the GNM-specific handler in StateBuilder."""
        # Create a minimal state dictionary
        minimal_state = {
            'n_asu': 2,
            'n_atoms_per_asu': 3,
            'n_cell': 3,
            'id_cell_ref': 0,
            'enm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Create a StateBuilder
        builder = StateBuilder(device=self.device)
        
        # Build a GNM from minimal state
        gnm = builder.build(TorchGNM, minimal_state)
        
        # Verify the GNM has the correct structure
        self.assertEqual(gnm.n_asu, 2)
        self.assertEqual(gnm.n_atoms_per_asu, 3)
        self.assertEqual(gnm.n_cell, 3)
        self.assertEqual(gnm.id_cell_ref, 0)
        
        # Verify crystal dictionary was created
        self.assertIsInstance(gnm.crystal, dict)
        self.assertTrue(callable(gnm.crystal.get('id_to_hkl')))
        self.assertTrue(callable(gnm.crystal.get('get_unitcell_origin')))

        # Test the methods in crystal dictionary
        hkl = gnm.crystal['id_to_hkl'](1)
        self.assertEqual(hkl, [1, 0, 0])

        origin = gnm.crystal['get_unitcell_origin']([1, 0, 0])
        self.assertIsInstance(origin, torch.Tensor)
        self.assertEqual(origin.shape, (3,))
        self.assertTrue(origin.requires_grad)

        # Verify asu_neighbors was created
        self.assertTrue(hasattr(gnm, 'asu_neighbors'))
        self.assertIsInstance(gnm.asu_neighbors, list)

    def _verify_gnm_state(self, state: Dict[str, Any]) -> None:
        """Verify the structure of the serialized GNM state."""
        # Check required attributes
        required_attrs = ['n_asu', 'n_atoms_per_asu', 'n_cell', 'id_cell_ref', 
                         'enm_cutoff', 'gamma_intra', 'gamma_inter']
        for attr in required_attrs:
            self.assertIn(attr, state, f"Missing required attribute: {attr}")
        
        # Check gamma array
        self.assertIn('gamma', state, "Missing gamma array")
        
        # Check asu_neighbors
        self.assertIn('asu_neighbors', state, "Missing asu_neighbors")
        
        # Check crystal
        self.assertIn('crystal', state, "Missing crystal")

    def _verify_torch_gnm(self, gnm: TorchGNM) -> None:
        """Verify the structure of the PyTorch GNM instance."""
        # Check required attributes
        required_attrs = ['n_asu', 'n_atoms_per_asu', 'n_cell', 'id_cell_ref', 'device']
        for attr in required_attrs:
            self.assertTrue(hasattr(gnm, attr), f"Missing required attribute: {attr}")

        # Check gamma tensor
        self.assertTrue(hasattr(gnm, 'gamma'), "Missing gamma tensor")
        if hasattr(gnm, 'gamma'):
            self.assertIsInstance(gnm.gamma, torch.Tensor, "gamma is not a tensor")

        # Check asu_neighbors
        self.assertTrue(hasattr(gnm, 'asu_neighbors'), "Missing asu_neighbors")

        # Check crystal
        self.assertTrue(hasattr(gnm, 'crystal'), "Missing crystal")

        # Check methods
        self.assertTrue(callable(getattr(gnm, 'compute_hessian', None)),
                       "compute_hessian method not found")
        self.assertTrue(callable(getattr(gnm, 'compute_K', None)),
                       "compute_K method not found")
        self.assertTrue(callable(getattr(gnm, 'compute_Kinv', None)),
                       "compute_Kinv method not found")

    def _print_state_structure(self, state: Dict[str, Any], indent: int = 0) -> None:
        """Print the structure of the state dictionary."""
        prefix = "  " * indent
        print(f"\nState Structure:")

        for key, value in state.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: Dict with {len(value)} keys")
                self._print_dict_structure(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: List with {len(value)} items")
                if len(value) > 0 and isinstance(value[0], dict):
                    self._print_dict_structure(value[0], indent + 1)
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: NumPy array with shape {value.shape} and dtype {value.dtype}")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")

    def _print_dict_structure(self, d: Dict[str, Any], indent: int = 0) -> None:
        """Print the structure of a dictionary."""
        prefix = "  " * indent

        # Handle non-string keys
        if any(not isinstance(k, str) for k in d.keys()):
            key_types = set(type(k).__name__ for k in d.keys())
            print(f"{prefix}Keys are of types: {', '.join(key_types)}")
            return

        # Print a sample of keys
        keys = list(d.keys())
        if len(keys) > 5:
            print(f"{prefix}Sample keys: {', '.join(keys[:5])}...")
        else:
            print(f"{prefix}Keys: {', '.join(keys)}")

if __name__ == '__main__':
    unittest.main()
