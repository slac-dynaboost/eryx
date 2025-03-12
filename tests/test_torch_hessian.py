import unittest
import numpy as np
import torch
from tests.torch_test_base import TorchComponentTestCase
from tests.test_helpers.component_tests import HessianTests

class TestTorchHessian(TorchComponentTestCase):
    """Test suite for PyTorch hessian calculation components."""
    
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
    
    def test_compute_gnm_hessian(self):
        """Test the compute_gnm_hessian method implementation."""
        # Create models with small test parameters for speed
        self.create_models()
        
        # Compute hessian with NumPy model
        np_hessian = self.np_model.gnm.compute_hessian()
        
        # Compute hessian with PyTorch model
        torch_hessian = self.torch_model.compute_gnm_hessian()
        
        # Compare structures using HessianTests utility
        structure_comparison = HessianTests.compare_hessian_structure(
            np_hessian, torch_hessian
        )
        
        # Check shape matches
        self.assertTrue(
            structure_comparison['shape_match'],
            f"Hessian shapes don't match: NP={structure_comparison['np_shape']}, "
            f"Torch={structure_comparison['torch_shape']}"
        )
        
        # Test value ranges
        self.assertTrue(
            structure_comparison['value_range_similar'],
            f"Hessian value ranges differ: "
            f"NP min/max={structure_comparison['np_min_abs']}/{structure_comparison['np_max_abs']}, "
            f"Torch min/max={structure_comparison['torch_min_abs']}/{structure_comparison['torch_max_abs']}"
        )
        
        # Test diagonal properties
        if structure_comparison['diag_real'] is not None:
            self.assertTrue(
                structure_comparison['diag_real'],
                "Diagonal elements should be real"
            )
            
            self.assertTrue(
                structure_comparison['diag_positive'],
                "Diagonal elements should be positive"
            )
    
    def test_compute_hessian(self):
        """Test the compute_hessian method implementation."""
        # Create models
        self.create_models()
        
        # Compute hessian with both models
        np_hessian = self.np_model.compute_hessian()
        torch_hessian = self.torch_model.compute_hessian()
        
        # Compare structures
        structure_comparison = HessianTests.compare_hessian_structure(
            np_hessian, torch_hessian
        )
        
        # Check shape matches
        self.assertTrue(
            structure_comparison['shape_match'],
            f"Hessian shapes don't match: NP={structure_comparison['np_shape']}, "
            f"Torch={structure_comparison['torch_shape']}"
        )
        
        # Test value ranges
        self.assertTrue(
            structure_comparison['value_range_similar'],
            f"Hessian value ranges differ: "
            f"NP min/max={structure_comparison['np_min_abs']}/{structure_comparison['np_max_abs']}, "
            f"Torch min/max={structure_comparison['torch_min_abs']}/{structure_comparison['torch_max_abs']}"
        )
        
        # Test several k-vectors (first, middle, last)
        k_dim = np_hessian.shape[2]
        test_indices = [0, k_dim // 2, k_dim - 1] if k_dim > 2 else [0]
        
        for k_idx in test_indices:
            with self.subTest(f"k_idx={k_idx}"):
                # Extract slice for this k-vector
                np_slice = np_hessian[:, :, k_idx, :, :]
                torch_slice = torch_hessian[:, :, k_idx, :, :].detach().cpu().numpy()
                
                # Compare with appropriate tolerance for complex projections
                self.assert_tensors_equal(
                    np_slice, torch_slice,
                    rtol=1e-4, atol=1e-7,
                    msg=f"Projected hessian values don't match for k_idx={k_idx}"
                )
    
    def test_hessian_dimensionality(self):
        """Test that hessian dimensionality matches between NumPy and PyTorch."""
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
        
        # Ensure k-vectors are built
        if not hasattr(np_model, 'kvec') or np_model.kvec is None:
            np_model._build_kvec_Brillouin()
            
        if not hasattr(torch_model, 'kvec') or torch_model.kvec is None:
            torch_model._build_kvec_Brillouin()
        
        # Compute hessian in both models
        np_hessian = np_model.compute_hessian()
        torch_hessian = torch_model.compute_hessian()
        
        # Convert PyTorch tensor to NumPy for shape comparison
        torch_hessian_shape = tuple(torch_hessian.shape)
        
        # Print diagnostic information
        print(f"NumPy hessian shape: {np_hessian.shape}")
        print(f"PyTorch hessian shape: {torch_hessian_shape}")
        
        # Calculate expected shape
        expected_kvectors = (params['hsampling'][2] + 1) * (params['ksampling'][2] + 1) * (params['lsampling'][2] + 1)
        print(f"Expected k-vectors: {expected_kvectors} ({params['hsampling'][2] + 1}×{params['ksampling'][2] + 1}×{params['lsampling'][2] + 1})")
        
        # Assert shapes match
        self.assertEqual(np_hessian.shape, torch_hessian_shape, 
                         f"Hessian shapes don't match: NP={np_hessian.shape}, Torch={torch_hessian_shape}")
        
        # Check if third dimension matches expected k-vector count 
        self.assertEqual(torch_hessian_shape[2], expected_kvectors,
                         f"Hessian third dimension should be {expected_kvectors} but got {torch_hessian_shape[2]}")
    
    def test_compute_gnm_K(self):
        """Test the compute_gnm_K method implementation."""
        # Create models
        self.create_models()
        
        # Compute full hessian first
        np_hessian = self.np_model.compute_hessian()
        torch_hessian = self.torch_model.compute_gnm_hessian()
        
        # Test with zero k-vector (simpler case)
        kvec = np.zeros(3)
        torch_kvec = torch.zeros(3, device=self.device)
        
        # Get K matrix for zero k-vector
        np_K = self.np_model.gnm.compute_K(np_hessian, kvec=kvec)
        torch_K = self.torch_model.compute_gnm_K(torch_hessian, kvec=torch_kvec)
        
        # Compare shapes and values
        np_shape = np_K.shape
        torch_shape = tuple(torch_K.shape)
        
        self.assertEqual(
            np_shape, torch_shape,
            f"K matrix shapes don't match: NP={np_shape}, Torch={torch_shape}"
        )
        
        # Compare values with appropriate tolerance
        self.assert_tensors_equal(
            np_K, torch_K,
            rtol=1e-4, atol=1e-7,
            msg="K matrix values don't match for zero k-vector"
        )
        
        # Test with non-zero k-vector
        kvec = np.array([0.1, 0.2, 0.3])
        torch_kvec = torch.tensor(kvec, device=self.device)
        
        # Get K matrix for non-zero k-vector
        np_K = self.np_model.gnm.compute_K(np_hessian, kvec=kvec)
        torch_K = self.torch_model.compute_gnm_K(torch_hessian, kvec=torch_kvec)
        
        # Compare values
        self.assert_tensors_equal(
            np_K, torch_K,
            rtol=1e-4, atol=1e-7,
            msg="K matrix values don't match for non-zero k-vector"
        )
    
    def test_compute_gnm_Kinv(self):
        """Test the compute_gnm_Kinv method implementation."""
        # Create models
        self.create_models()
        
        # Compute full hessian first
        np_hessian = self.np_model.compute_hessian()
        torch_hessian = self.torch_model.compute_gnm_hessian()
        
        # Test with zero k-vector
        kvec = np.zeros(3)
        torch_kvec = torch.zeros(3, device=self.device)
        
        # Get Kinv matrix for zero k-vector
        np_Kinv = self.np_model.gnm.compute_Kinv(np_hessian, kvec=kvec, reshape=True)
        torch_Kinv = self.torch_model.compute_gnm_Kinv(torch_hessian, kvec=torch_kvec, reshape=True)
        
        # Compare shapes and values
        np_shape = np_Kinv.shape
        torch_shape = tuple(torch_Kinv.shape)
        
        self.assertEqual(
            np_shape, torch_shape,
            f"Kinv matrix shapes don't match: NP={np_shape}, Torch={torch_shape}"
        )
        
        # Compare values with appropriate tolerance
        self.assert_tensors_equal(
            np_Kinv, torch_Kinv,
            rtol=1e-3, atol=1e-6,  # Use slightly higher tolerance for inverse
            msg="Kinv matrix values don't match for zero k-vector"
        )

if __name__ == '__main__':
    unittest.main()
