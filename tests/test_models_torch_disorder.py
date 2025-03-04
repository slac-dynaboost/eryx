import unittest
import os
import torch
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from unittest.mock import patch, MagicMock

class TestOnePhononDisorder(unittest.TestCase):
    def setUp(self):
        # Set up the testing framework
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-3, atol=1e-5)
        
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
        
        # Log file prefix for ground truth data
        self.apply_disorder_log = "logs/eryx.models.apply_disorder"
        
        # Ensure log file exists
        log_file_path = f"{self.apply_disorder_log}.log"
        self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
        
        # Create minimal test model with mock methods and properties
        self.model = self._create_test_model()
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes and mock methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes
        model.device = self.device
        model.batch_size = 1000
        model.n_asu = 2
        model.n_dof_per_asu = 6
        model.n_dof_per_asu_actual = 12  # 4 atoms * 3 dimensions
        model.n_cell = 3
        model.hsampling = (0, 5, 3)
        model.ksampling = (0, 5, 3)
        model.lsampling = (0, 5, 3)
        
        # Create small test grid
        grid_size = 100
        model.q_grid = torch.rand((grid_size, 3), device=self.device)
        model.res_mask = torch.ones(grid_size, dtype=torch.bool, device=self.device)
        model.res_mask[:10] = False  # Some points outside resolution mask
        
        # Mock tensors for structure factor calculation
        model.model = MagicMock()
        model.model.xyz = [torch.rand((4, 3), device=self.device) for _ in range(model.n_asu)]
        model.model.ff_a = [torch.rand((4, 4), device=self.device) for _ in range(model.n_asu)]
        model.model.ff_b = [torch.rand((4, 4), device=self.device) for _ in range(model.n_asu)]
        model.model.ff_c = [torch.rand((4), device=self.device) for _ in range(model.n_asu)]
        model.model.adp = [torch.ones(4, device=self.device)]
        
        # Mock ADP tensor
        model.ADP = torch.ones(4, device=self.device)
        
        # Mock Amat tensor (projection matrix)
        model.Amat = [torch.rand((12, 6), device=self.device) for _ in range(model.n_asu)]
        
        # Mock V (eigenvectors) and Winv (inverse eigenvalues)
        model.V = torch.rand(
            (model.hsampling[2], model.ksampling[2], model.lsampling[2], 
            model.n_asu * model.n_dof_per_asu, model.n_asu * model.n_dof_per_asu),
            dtype=torch.complex64,
            device=self.device
        )
        model.Winv = torch.rand(
            (model.hsampling[2], model.ksampling[2], model.lsampling[2], 
            model.n_asu * model.n_dof_per_asu),
            dtype=torch.complex64,
            device=self.device
        )
        
        # Mock _at_kvec_from_miller_points method
        def mock_at_kvec_from_miller_points(hkl_kvec):
            # Return a subset of indices for testing
            indices = torch.arange(20, 50, device=self.device)
            return indices
        model._at_kvec_from_miller_points = mock_at_kvec_from_miller_points
        
        # Mock structure_factors with a patch
        self.structure_factors_patch = patch('eryx.scatter_torch.structure_factors')
        self.mock_structure_factors = self.structure_factors_patch.start()
        
        # Configure mock structure_factors to return a tensor
        def mock_sf(*args, **kwargs):
            batch_size = args[0].shape[0]
            if kwargs.get('compute_qF', False):
                # Return structure factors with components
                return torch.complex(
                    torch.rand((batch_size, 6), device=self.device),
                    torch.rand((batch_size, 6), device=self.device)
                )
            else:
                # Return simple structure factors
                return torch.complex(
                    torch.rand(batch_size, device=self.device),
                    torch.rand(batch_size, device=self.device)
                )
        self.mock_structure_factors.side_effect = mock_sf
        
        return model
    
    def tearDown(self):
        # Stop patches
        self.structure_factors_patch.stop()
    
    def test_apply_disorder_shape(self):
        """Test that the apply_disorder method returns the correct shape."""
        # Run the method
        Id = self.model.apply_disorder()
        
        # Check shape and dtype
        self.assertEqual(Id.shape, (self.model.q_grid.shape[0],))
        self.assertTrue(torch.is_floating_point(Id))
        
        # Check NaN values where res_mask is False
        self.assertTrue(torch.all(torch.isnan(Id[~self.model.res_mask])))
        
        # Check non-NaN values where res_mask is True
        self.assertFalse(torch.any(torch.isnan(Id[self.model.res_mask])))
    
    def test_apply_disorder_rank_selection(self):
        """Test that rank selection works correctly."""
        # Run with all modes (rank=-1)
        Id_all = self.model.apply_disorder(rank=-1)
        
        # Run with specific mode (rank=0)
        Id_single = self.model.apply_disorder(rank=0)
        
        # Shapes should be the same
        self.assertEqual(Id_all.shape, Id_single.shape)
        
        # Values should be different
        # We only compare points within resolution mask
        mask = self.model.res_mask
        self.assertFalse(torch.allclose(
            Id_all[mask], Id_single[mask], 
            rtol=1e-3, atol=1e-5
        ))
    
    def test_apply_disorder_gradient_flow(self):
        """Test gradient flow through apply_disorder."""
        # Enable anomaly detection to help debug gradient issues
        torch.autograd.set_detect_anomaly(True)
        
        # Make sure tensors require gradients
        self.model.q_grid.requires_grad_(True)
        self.model.V.requires_grad_(True)
        self.model.Winv.requires_grad_(True)
        
        # Configure mock structure_factors to return a tensor with gradients
        def mock_sf_with_grad(*args, **kwargs):
            batch_size = args[0].shape[0]
            if kwargs.get('compute_qF', False):
                # Return structure factors with components that depend on q_grid
                q_grid = args[0]  # This is the q_grid tensor that needs gradients
                real_part = torch.sin(torch.sum(q_grid, dim=1)).unsqueeze(1).expand(batch_size, 6)
                imag_part = torch.cos(torch.sum(q_grid, dim=1)).unsqueeze(1).expand(batch_size, 6)
                return torch.complex(real_part, imag_part)
            else:
                # Return simple structure factors that depend on q_grid
                q_grid = args[0]
                real_part = torch.sin(torch.sum(q_grid, dim=1))
                imag_part = torch.cos(torch.sum(q_grid, dim=1))
                return torch.complex(real_part, imag_part)
        self.mock_structure_factors.side_effect = mock_sf_with_grad
        
        # Run the method
        Id = self.model.apply_disorder()
        
        # Create a scalar loss from the output
        # Only use points within the resolution mask
        mask = self.model.res_mask
        loss = Id[mask].mean()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients flowed to input parameters
        self.assertIsNotNone(self.model.q_grid.grad)
        self.assertIsNotNone(self.model.V.grad)
        self.assertIsNotNone(self.model.Winv.grad)
        
        # Verify gradients are not all zeros
        self.assertFalse(torch.allclose(self.model.q_grid.grad, torch.zeros_like(self.model.q_grid.grad)))
        self.assertFalse(torch.allclose(self.model.V.grad, torch.zeros_like(self.model.V.grad)))
        self.assertFalse(torch.allclose(self.model.Winv.grad, torch.zeros_like(self.model.Winv.grad)))
        
        # Disable anomaly detection after test
        torch.autograd.set_detect_anomaly(False)
    
    def test_apply_disorder_adp_selection(self):
        """Test that ADP selection works correctly."""
        # Set different values for model.adp and self.ADP
        self.model.ADP = torch.ones(4, device=self.device) * 2.0
        self.model.model.adp = [torch.ones(4, device=self.device) * 5.0]
        
        # Run with computed ADPs
        Id_computed = self.model.apply_disorder(use_data_adp=False)
        
        # Run with data ADPs
        Id_data = self.model.apply_disorder(use_data_adp=True)
        
        # Values should be different due to different ADPs
        mask = self.model.res_mask
        self.assertFalse(torch.allclose(
            Id_computed[mask], Id_data[mask], 
            rtol=1e-3, atol=1e-5
        ))
    
    def test_apply_disorder_ground_truth(self):
        """Test against ground truth data."""
        # Skip this test if the log file doesn't exist
        log_file_path = f"{self.apply_disorder_log}.log"
        if not os.path.exists(log_file_path):
            self.skipTest(f"Ground truth data not found: {log_file_path}")
            
        # Load the ground truth data
        logs = self.logger.loadLog(log_file_path)
        if not logs or len(logs) < 2:  # Need at least one input/output pair
            self.skipTest("Insufficient ground truth data in log file")
            
        # Get the input data from the first log entry
        input_args = self.logger.serializer.deserialize(logs[0]['args'])
        expected_output = self.logger.serializer.deserialize(logs[0]['result'])
        
        # Extract parameters from the input data
        # Note: The exact structure depends on how apply_disorder was logged
        rank = input_args[0] if len(input_args) > 0 else -1
        outdir = input_args[1] if len(input_args) > 1 else None
        use_data_adp = input_args[2] if len(input_args) > 2 else False
        
        # Run the apply_disorder method with the same parameters
        actual_output = self.model.apply_disorder(rank=rank, outdir=outdir, use_data_adp=use_data_adp)
        
        # Convert the PyTorch tensor to NumPy for comparison with ground truth
        actual_output_np = actual_output.detach().cpu().numpy()
        
        # Compare with expected output
        # We need to handle NaN values specially
        if isinstance(expected_output, np.ndarray):
            # Create masks for non-NaN values in both arrays
            expected_mask = ~np.isnan(expected_output)
            actual_mask = ~np.isnan(actual_output_np)
            
            # Check that NaN positions match
            self.assertTrue(np.array_equal(expected_mask, actual_mask),
                           "NaN positions don't match between expected and actual outputs")
            
            # Compare only non-NaN values
            if np.any(expected_mask):
                np.testing.assert_allclose(
                    expected_output[expected_mask],
                    actual_output_np[expected_mask],
                    rtol=1e-3, atol=1e-5,
                    err_msg="Output values don't match ground truth"
                )
                
            # Check shapes match
            self.assertEqual(expected_output.shape, actual_output_np.shape,
                           "Output shape doesn't match ground truth")
        else:
            self.fail(f"Expected output is not a NumPy array: {type(expected_output)}")
            
        # Alternative approach using the TorchTesting framework
        self.assertTrue(
            self.torch_testing.testTorchCallable(self.apply_disorder_log, self.model.apply_disorder),
            "apply_disorder failed ground truth test"
        )

if __name__ == '__main__':
    unittest.main()
