import os
import unittest
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon

class TestKvectorMethods(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
    def create_models(self, test_params=None):
        """Create NumPy and PyTorch models for comparative testing."""
        # Import NumPy model for comparison
        from eryx.models import OnePhonon as NumpyOnePhonon
        
        # Default test parameters
        self.test_params = test_params or {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Create NumPy model for reference
        self.np_model = NumpyOnePhonon(**self.test_params)
        
        # Create PyTorch model
        self.torch_model = OnePhonon(
            **self.test_params,
            device=self.device
        )

    def test_center_kvec(self):
        """Test the _center_kvec method implementation by direct comparison."""
        # Create models for comparison
        self.create_models()
        
        # Test cases for _center_kvec
        test_cases = [
            (0, 2), (1, 2),        # Even L
            (0, 3), (1, 3), (2, 3), # Odd L
            (5, 10), (9, 10),      # Larger L
            (50, 100), (99, 100)    # Much larger L
        ]
        
        for x, L in test_cases:
            with self.subTest(f"x={x}, L={L}"):
                np_result = self.np_model._center_kvec(x, L)
                torch_result = self.torch_model._center_kvec(x, L)
                
                # Convert torch result to Python scalar if needed
                if isinstance(torch_result, torch.Tensor):
                    torch_result = torch_result.item()
                
                # Compare results - must be exactly equal
                self.assertEqual(np_result, torch_result,
                               f"_center_kvec returns different values: NP={np_result}, Torch={torch_result}")
#        
    def test_at_kvec_from_miller_points(self):
        """Test the _at_kvec_from_miller_points method against NumPy implementation."""
        # Create models for comparison
        self.create_models()
        
        # Test different miller points
        test_points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        
        for point in test_points:
            # Call method on both implementations
            np_indices = self.np_model._at_kvec_from_miller_points(point)
            torch_indices = self.torch_model._at_kvec_from_miller_points(point)
            
            # Convert to NumPy arrays for comparison
            if isinstance(torch_indices, torch.Tensor):
                torch_indices = torch_indices.cpu().numpy()
            if not isinstance(np_indices, np.ndarray):
                np_indices = np.array(np_indices)
            
            # Compare results
            np.testing.assert_array_equal(np_indices, torch_indices,
                                       f"Indices don't match for miller point {point}")
#
    def test_kvec_brillouin_equivalence(self):
        """Compare kvec and kvec_norm after _build_kvec_Brillouin."""
        # Import TensorComparison for precise tensor comparison
        from tests.torch_test_utils import TensorComparison
        
        # Create models for comparison
        self.create_models()
        
        # _build_kvec_Brillouin is called during __init__
        
        # Compare kvec (should be numerically identical float64)
        TensorComparison.assert_tensors_equal(
            self.np_model.kvec, self.torch_model.kvec,
            rtol=1e-9, atol=1e-12, # Use very tight tolerance
            msg="kvec comparison failed"
        )
        
        # Compare kvec_norm (should be numerically identical float64)
        TensorComparison.assert_tensors_equal(
            self.np_model.kvec_norm, self.torch_model.kvec_norm,
            rtol=1e-9, atol=1e-12, # Use very tight tolerance
            msg="kvec_norm comparison failed"
        )
            
    def test_log_completeness(self):
        """Verify k-vector method logs exist and contain required attributes."""
        if not hasattr(self, 'verify_logs') or not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify k-vector method logs
        self.verify_required_logs(self.module_name, "_build_kvec_Brillouin", ["kvec", "kvec_norm"])
        self.verify_required_logs(self.module_name, "_center_kvec", [])
        self.verify_required_logs(self.module_name, "_at_kvec_from_miller_points", [])
#
class TestOnePhononKvector(TestKvectorMethods):
    """Legacy class for backward compatibility."""
    
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Initialize test_params
        self.test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }

if __name__ == '__main__':
    unittest.main()
