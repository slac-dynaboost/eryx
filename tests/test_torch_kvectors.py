import unittest
import numpy as np
import torch
from tests.torch_test_base import TorchComponentTestCase
from tests.test_helpers.component_tests import KVectorTests

class TestTorchKVectors(TorchComponentTestCase):
    """Test suite for PyTorch k-vector construction components."""
    
    def setUp(self):
        """Set up test environment with default configuration."""
        super().setUp()
        # Use smaller test parameters for faster testing
        self.test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],  # Smaller grid for faster testing
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    
    def test_center_kvec(self):
        """Test the _center_kvec method implementation."""
        # Create models
        self.create_models()
        
        # Test cases for _center_kvec
        test_cases = [
            (0, 2), (1, 2),  # Even L
            (0, 3), (1, 3), (2, 3),  # Odd L
            (5, 10), (9, 10),  # Larger L
            (50, 100), (99, 100)  # Much larger L
        ]
        
        for x, L in test_cases:
            with self.subTest(f"x={x}, L={L}"):
                np_result = self.np_model._center_kvec(x, L)
                torch_result = self.torch_model._center_kvec(x, L)
                
                # Convert torch result to Python scalar if needed
                if isinstance(torch_result, torch.Tensor):
                    torch_result = torch_result.item()
                
                self.assertEqual(
                    np_result, torch_result,
                    f"_center_kvec returns different values: NP={np_result}, Torch={torch_result}"
                )
    
    def test_build_kvec_brillouin(self):
        """Test the _build_kvec_Brillouin method implementation."""
        # Create models with various sampling parameters
        test_cases = [
            {'hsampling': [-2, 2, 2], 'ksampling': [-2, 2, 2], 'lsampling': [-2, 2, 2]},
            {'hsampling': [-1, 1, 3], 'ksampling': [-1, 1, 3], 'lsampling': [-1, 1, 3]},
            {'hsampling': [-3, 3, 1], 'ksampling': [-3, 3, 1], 'lsampling': [-3, 3, 1]}
        ]
        
        for i, params in enumerate(test_cases):
            with self.subTest(f"case_{i+1}"):
                # Update test parameters
                test_params = self.test_params.copy()
                test_params.update(params)
                
                # Create models with these parameters
                self.create_models(test_params)
                
                # Run _build_kvec_Brillouin on both models
                self.np_model._build_kvec_Brillouin()
                self.torch_model._build_kvec_Brillouin()
                
                # Compare kvec tensors
                self.assert_tensors_equal(
                    self.np_model.kvec, 
                    self.torch_model.kvec,
                    rtol=1e-5, atol=1e-8,
                    msg=f"kvec values don't match for case {i+1}"
                )
                
                # Compare kvec_norm tensors
                self.assert_tensors_equal(
                    self.np_model.kvec_norm, 
                    self.torch_model.kvec_norm,
                    rtol=1e-5, atol=1e-8,
                    msg=f"kvec_norm values don't match for case {i+1}"
                )
    
    def test_at_kvec_from_miller_points(self):
        """Test the _at_kvec_from_miller_points method implementation."""
        # Create models
        self.create_models()
        
        # Test with various miller points
        test_points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        
        for point in test_points:
            with self.subTest(f"miller_point={point}"):
                # Get indices from both implementations
                np_indices = self.np_model._at_kvec_from_miller_points(point)
                torch_indices = self.torch_model._at_kvec_from_miller_points(point)
                
                # Convert torch result to NumPy if needed
                if isinstance(torch_indices, torch.Tensor):
                    torch_indices_array = torch_indices.cpu().numpy()
                else:
                    torch_indices_array = np.array(torch_indices)
                
                # Convert np result to NumPy if needed
                if not isinstance(np_indices, np.ndarray):
                    np_indices_array = np.array(np_indices)
                else:
                    np_indices_array = np_indices
                
                # They should match exactly (no tolerance needed for indices)
                np.testing.assert_array_equal(
                    np_indices_array, torch_indices_array,
                    err_msg=f"Indices don't match for miller point {point}"
                )
    
    def test_kvec_dimensions(self):
        """Test that kvec and kvec_norm have correct dimensions after initialization."""
        # Create models
        self.create_models()
        
        # Verify kvec dimensions match sampling parameters
        h_dim = self.test_params['hsampling'][2]
        k_dim = self.test_params['ksampling'][2]
        l_dim = self.test_params['lsampling'][2]
        
        # Check NumPy dimensions
        self.assertEqual(
            self.np_model.kvec.shape,
            (h_dim, k_dim, l_dim, 3),
            f"NumPy kvec shape incorrect: expected {(h_dim, k_dim, l_dim, 3)}, got {self.np_model.kvec.shape}"
        )
        
        # Check PyTorch dimensions
        torch_shape = tuple(self.torch_model.kvec.shape)
        self.assertEqual(
            torch_shape,
            (h_dim, k_dim, l_dim, 3),
            f"PyTorch kvec shape incorrect: expected {(h_dim, k_dim, l_dim, 3)}, got {torch_shape}"
        )

if __name__ == '__main__':
    unittest.main()
