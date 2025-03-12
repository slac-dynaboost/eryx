import unittest
import numpy as np
import torch
from tests.torch_test_base import TorchComponentTestCase
from tests.test_helpers.component_tests import KVectorTests

class TestTorchKVector(TorchComponentTestCase):
    """
    Test case for k-vector construction in PyTorch implementation.
    
    This test validates that the PyTorch implementation correctly calculates
    k-vectors with the same dimensionality as the NumPy implementation.
    """
    
    def test_kvector_dimensionality(self):
        """Test that k-vector dimensionality matches between NumPy and PyTorch."""
        # Create models with test parameters
        params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],  # Should produce 3×3×3=27 k-vectors
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        np_model, torch_model = self.create_models(params)
        
        # Build k-vectors in both models
        np_model._build_kvec_Brillouin()
        torch_model._build_kvec_Brillouin()
        
        # Check dimensionality
        np_kvec_shape = np_model.kvec.shape
        torch_kvec_shape = tuple(torch_model.kvec.shape)
        
        # Print diagnostic information
        print(f"NumPy kvec shape: {np_kvec_shape}")
        print(f"PyTorch kvec shape: {torch_kvec_shape}")
        
        # Assert shapes match
        self.assertEqual(np_kvec_shape, torch_kvec_shape, 
                         f"K-vector shapes don't match: NP={np_kvec_shape}, Torch={torch_kvec_shape}")
        
        # Verify interpretation of sampling parameter
        expected_points_per_dim = params['hsampling'][2] + 1  # n+1 for sampling parameter n
        expected_shape = (expected_points_per_dim, expected_points_per_dim, expected_points_per_dim, 3)
        self.assertEqual(torch_kvec_shape, expected_shape,
                         f"Expected {expected_shape} points (sampling+1) but got {torch_kvec_shape}")
        
        # Also check k-vector values (few samples)
        for h_idx in [0, 1]:
            for k_idx in [0, 1]:
                for l_idx in [0, 1]:
                    np_value = np_model.kvec[h_idx, k_idx, l_idx]
                    torch_value = torch_model.kvec[h_idx, k_idx, l_idx].detach().cpu().numpy()
                    
                    # Compare with appropriate tolerance
                    np.testing.assert_allclose(
                        np_value, torch_value, 
                        rtol=1e-5, atol=1e-8,
                        err_msg=f"K-vector values at ({h_idx},{k_idx},{l_idx}) don't match"
                    )
    
    def test_sampling_parameter_interpretation(self):
        """Test different sampling parameters to confirm consistent interpretation."""
        # Test with different sampling parameters
        for sampling in [1, 2, 3, 4]:
            params = {
                'pdb_path': 'tests/pdbs/5zck_p1.pdb',
                'hsampling': [-2, 2, sampling],
                'ksampling': [-2, 2, sampling],
                'lsampling': [-2, 2, sampling],
                'expand_p1': True,
                'res_limit': 0.0,
                'gnm_cutoff': 4.0,
                'gamma_intra': 1.0,
                'gamma_inter': 1.0
            }
            np_model, torch_model = self.create_models(params)
            
            # Build k-vectors in both models
            np_model._build_kvec_Brillouin()
            torch_model._build_kvec_Brillouin()
            
            # Check dimensionality
            np_kvec_shape = np_model.kvec.shape
            torch_kvec_shape = tuple(torch_model.kvec.shape)
            
            # Expected shape based on sampling+1 formula
            expected_points_per_dim = sampling + 1
            expected_shape = (expected_points_per_dim, expected_points_per_dim, expected_points_per_dim, 3)
            
            # Assert shapes match both NumPy and expected formula
            self.assertEqual(np_kvec_shape, torch_kvec_shape, 
                             f"With sampling={sampling}, k-vector shapes don't match: NP={np_kvec_shape}, Torch={torch_kvec_shape}")
            self.assertEqual(torch_kvec_shape, expected_shape,
                             f"With sampling={sampling}, expected {expected_shape} but got {torch_kvec_shape}")
            
            print(f"Sampling={sampling}: NumPy={np_kvec_shape}, PyTorch={torch_kvec_shape}, Expected={expected_shape}")

if __name__ == '__main__':
    unittest.main()
