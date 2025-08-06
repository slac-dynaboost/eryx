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

    # Removed test_kvec_brillouin_equivalence because the kvec attribute
    # is intentionally different between grid mode (BZ sampling) and arbitrary-q mode
    # (direct k=q/2pi). Comparing them directly is not meaningful.
    # The internal logic of _build_kvec_Brillouin for grid mode is tested implicitly
    # by downstream methods like compute_gnm_phonons and compute_covariance_matrix.

    def test_log_completeness(self):
        """Verify k-vector method logs exist and contain required attributes."""
        if not hasattr(self, 'verify_logs') or not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify k-vector method logs
        self.verify_required_logs(self.module_name, "_build_kvec_Brillouin", ["kvec", "kvec_norm"])
        self.verify_required_logs(self.module_name, "_center_kvec", [])
        self.verify_required_logs(self.module_name, "_at_kvec_from_miller_points", [])
#
class TestTorchKVector(TestBase):
    """Test the indexing functions in the PyTorch implementation."""
    
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Create a minimal model for testing indexing functions
        self.model = self._create_minimal_model()
    
    def _create_minimal_model(self):
        """Create a minimal model with just the attributes needed for indexing tests."""
        from eryx.models_torch import OnePhonon
        
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set basic attributes needed for indexing functions
        model.device = self.device
        model.hsampling = [-2, 2, 2]  # min, max, sampling
        model.ksampling = [-3, 3, 3]  # min, max, sampling
        model.lsampling = [-4, 4, 4]  # min, max, sampling
        
        # Calculate map_shape based on sampling parameters
        h_steps = int(model.hsampling[2] * (model.hsampling[1] - model.hsampling[0]) + 1)
        k_steps = int(model.ksampling[2] * (model.ksampling[1] - model.ksampling[0]) + 1)
        l_steps = int(model.lsampling[2] * (model.lsampling[1] - model.lsampling[0]) + 1)
        model.map_shape = (h_steps, k_steps, l_steps)
        
        # Set use_arbitrary_q flag to False for grid-based mode
        model.use_arbitrary_q = False
        
        return model
    
    def test_flat_to_3d_indices(self):
        """Test conversion from flat indices to 3D indices."""
        # Test cases: flat_idx, expected (h, k, l)
        test_cases = [
            (0, (0, 0, 0)),  # Origin
            (1, (0, 0, 1)),  # Next in l dimension
            (self.model.map_shape[2], (0, 1, 0)),  # Next in k dimension
            (self.model.map_shape[1] * self.model.map_shape[2], (1, 0, 0)),  # Next in h dimension
            (self.model.map_shape[1] * self.model.map_shape[2] - 1, (0, self.model.map_shape[1]-1, self.model.map_shape[2]-1)),  # Last in first h-plane
            (self.model.map_shape[0] * self.model.map_shape[1] * self.model.map_shape[2] - 1, 
             (self.model.map_shape[0]-1, self.model.map_shape[1]-1, self.model.map_shape[2]-1))  # Last element
        ]
        
        for flat_idx, expected in test_cases:
            with self.subTest(f"flat_idx={flat_idx}, expected={expected}"):
                # Convert to tensor
                flat_tensor = torch.tensor([flat_idx], device=self.device, dtype=torch.long)
                
                # Call the method
                h_indices, k_indices, l_indices = self.model._flat_to_3d_indices(flat_tensor)
                
                # Convert to tuples for comparison
                result = (h_indices.item(), k_indices.item(), l_indices.item())
                
                # Assert equality
                self.assertEqual(result, expected, 
                               f"flat_to_3d_indices failed: got {result}, expected {expected}")
    
    def test_3d_to_flat_indices(self):
        """Test conversion from 3D indices to flat indices."""
        # Test cases: (h, k, l), expected flat_idx
        test_cases = [
            ((0, 0, 0), 0),  # Origin
            ((0, 0, 1), 1),  # Next in l dimension
            ((0, 1, 0), self.model.map_shape[2]),  # Next in k dimension
            ((1, 0, 0), self.model.map_shape[1] * self.model.map_shape[2]),  # Next in h dimension
            ((self.model.map_shape[0]-1, self.model.map_shape[1]-1, self.model.map_shape[2]-1), 
             self.model.map_shape[0] * self.model.map_shape[1] * self.model.map_shape[2] - 1)  # Last element
        ]
        
        for indices, expected in test_cases:
            with self.subTest(f"indices={indices}, expected={expected}"):
                # Convert to tensors
                h_tensor = torch.tensor([indices[0]], device=self.device, dtype=torch.long)
                k_tensor = torch.tensor([indices[1]], device=self.device, dtype=torch.long)
                l_tensor = torch.tensor([indices[2]], device=self.device, dtype=torch.long)
                
                # Call the method
                flat_indices = self.model._3d_to_flat_indices(h_tensor, k_tensor, l_tensor)
                
                # Get result
                result = flat_indices.item()
                
                # Assert equality
                self.assertEqual(result, expected, 
                               f"3d_to_flat_indices failed: got {result}, expected {expected}")
    
    def test_flat_to_3d_indices_bz(self):
        """Test conversion from flat indices to 3D indices in Brillouin zone."""
        # Get BZ dimensions
        h_dim_bz = int(self.model.hsampling[2])
        k_dim_bz = int(self.model.ksampling[2])
        l_dim_bz = int(self.model.lsampling[2])
        
        # Test cases: flat_idx, expected (h, k, l)
        test_cases = [
            (0, (0, 0, 0)),  # Origin
            (1, (0, 0, 1)),  # Next in l dimension
            (l_dim_bz, (0, 1, 0)),  # Next in k dimension
            (k_dim_bz * l_dim_bz, (1, 0, 0)),  # Next in h dimension
            (h_dim_bz * k_dim_bz * l_dim_bz - 1, 
             (h_dim_bz-1, k_dim_bz-1, l_dim_bz-1))  # Last element
        ]
        
        for flat_idx, expected in test_cases:
            with self.subTest(f"flat_idx={flat_idx}, expected={expected}"):
                # Convert to tensor
                flat_tensor = torch.tensor([flat_idx], device=self.device, dtype=torch.long)
                
                # Call the method
                h_indices, k_indices, l_indices = self.model._flat_to_3d_indices_bz(flat_tensor)
                
                # Convert to tuples for comparison
                result = (h_indices.item(), k_indices.item(), l_indices.item())
                
                # Assert equality
                self.assertEqual(result, expected, 
                               f"flat_to_3d_indices_bz failed: got {result}, expected {expected}")
    
    def test_3d_to_flat_indices_bz(self):
        """Test conversion from 3D indices to flat indices in Brillouin zone."""
        # Get BZ dimensions
        h_dim_bz = int(self.model.hsampling[2])
        k_dim_bz = int(self.model.ksampling[2])
        l_dim_bz = int(self.model.lsampling[2])
        
        # Test cases: (h, k, l), expected flat_idx
        test_cases = [
            ((0, 0, 0), 0),  # Origin
            ((0, 0, 1), 1),  # Next in l dimension
            ((0, 1, 0), l_dim_bz),  # Next in k dimension
            ((1, 0, 0), k_dim_bz * l_dim_bz),  # Next in h dimension
            ((h_dim_bz-1, k_dim_bz-1, l_dim_bz-1), 
             h_dim_bz * k_dim_bz * l_dim_bz - 1)  # Last element
        ]
        
        for indices, expected in test_cases:
            with self.subTest(f"indices={indices}, expected={expected}"):
                # Convert to tensors
                h_tensor = torch.tensor([indices[0]], device=self.device, dtype=torch.long)
                k_tensor = torch.tensor([indices[1]], device=self.device, dtype=torch.long)
                l_tensor = torch.tensor([indices[2]], device=self.device, dtype=torch.long)
                
                # Call the method
                flat_indices = self.model._3d_to_flat_indices_bz(h_tensor, k_tensor, l_tensor)
                
                # Get result
                result = flat_indices.item()
                
                # Assert equality
                self.assertEqual(result, expected, 
                               f"3d_to_flat_indices_bz failed: got {result}, expected {expected}")
    
    def test_batch_indexing(self):
        """Test batch processing of indices."""
        # Create batch of flat indices
        flat_indices = torch.tensor([0, 1, 2, 10, 20], device=self.device, dtype=torch.long)
        
        # Test batch conversion to 3D indices
        h_indices, k_indices, l_indices = self.model._flat_to_3d_indices(flat_indices)
        
        # Test batch conversion back to flat indices
        flat_indices_back = self.model._3d_to_flat_indices(h_indices, k_indices, l_indices)
        
        # Verify round-trip conversion
        torch.testing.assert_close(flat_indices, flat_indices_back, 
                                  msg="Batch round-trip conversion failed")
        
        # Same for BZ indices
        h_indices_bz, k_indices_bz, l_indices_bz = self.model._flat_to_3d_indices_bz(flat_indices)
        flat_indices_bz_back = self.model._3d_to_flat_indices_bz(h_indices_bz, k_indices_bz, l_indices_bz)
        
        # Verify BZ round-trip conversion
        torch.testing.assert_close(flat_indices, flat_indices_bz_back, 
                                  msg="BZ batch round-trip conversion failed")
    
    def test_arbitrary_q_mode(self):
        """Test indexing functions in arbitrary q-vector mode."""
        # Create a model in arbitrary q-vector mode
        model = self._create_minimal_model()
        model.use_arbitrary_q = True
        
        # Create sample q-vectors
        q_vectors = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], device=self.device)
        model.q_vectors = q_vectors
        
        # Test flat_to_3d_indices in arbitrary mode (should be identity)
        flat_indices = torch.tensor([0, 1, 2], device=self.device, dtype=torch.long)
        h_indices, k_indices, l_indices = model._flat_to_3d_indices(flat_indices)
        
        # All should be the same as input in arbitrary mode
        torch.testing.assert_close(flat_indices, h_indices, 
                                  msg="flat_to_3d_indices should return input for h in arbitrary mode")
        torch.testing.assert_close(flat_indices, k_indices, 
                                  msg="flat_to_3d_indices should return input for k in arbitrary mode")
        torch.testing.assert_close(flat_indices, l_indices, 
                                  msg="flat_to_3d_indices should return input for l in arbitrary mode")
        
        # Test 3d_to_flat_indices in arbitrary mode (should be identity for h_indices)
        flat_indices_back = model._3d_to_flat_indices(h_indices, k_indices, l_indices)
        torch.testing.assert_close(flat_indices, flat_indices_back, 
                                  msg="3d_to_flat_indices should return h_indices in arbitrary mode")

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
