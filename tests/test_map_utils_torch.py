import unittest
import os
import torch
import numpy as np
from eryx.map_utils_torch import generate_grid, compute_resolution, get_resolution_mask
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping

class TestMapUtilsTorch(unittest.TestCase):
    def setUp(self):
        # Set up the testing framework
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-5, atol=1e-8)
        
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
        
        # Log file prefixes for ground truth data
        self.generate_grid_log = "logs/eryx.map_utils.generate_grid"
        self.compute_resolution_log = "logs/eryx.map_utils.compute_resolution"
        self.get_resolution_mask_log = "logs/eryx.map_utils.get_resolution_mask"
        
        # Ensure log files exist
        for log_file in [self.generate_grid_log, self.compute_resolution_log, self.get_resolution_mask_log]:
            log_file_path = f"{log_file}.log"
            self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
    
    def test_generate_grid(self):
        """Test generate_grid against ground truth data."""
        # Test against ground truth using TorchTesting
        self.assertTrue(
            self.torch_testing.testTorchCallable(self.generate_grid_log, generate_grid),
            "generate_grid failed ground truth test"
        )
        
        # Additional test for gradient flow
        A_inv = torch.eye(3, requires_grad=True)
        hsampling = (-3, 3, 1)
        ksampling = (-3, 3, 1)
        lsampling = (-3, 3, 1)
        
        q_grid, _ = generate_grid(A_inv, hsampling, ksampling, lsampling)
        loss = q_grid.sum()
        loss.backward()
        
        self.assertIsNotNone(A_inv.grad)
        self.assertFalse(torch.allclose(A_inv.grad, torch.zeros_like(A_inv.grad)))
    
    def test_compute_resolution(self):
        """Test compute_resolution against ground truth data."""
        # Test against ground truth using TorchTesting
        self.assertTrue(
            self.torch_testing.testTorchCallable(self.compute_resolution_log, compute_resolution),
            "compute_resolution failed ground truth test"
        )
        
        # Additional test for gradient flow
        cell = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], requires_grad=True)
        hkl = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        resolution = compute_resolution(cell, hkl)
        loss = resolution.sum()
        loss.backward()
        
        self.assertIsNotNone(cell.grad)
        self.assertFalse(torch.allclose(cell.grad, torch.zeros_like(cell.grad)))
    
    def test_get_resolution_mask(self):
        """Test get_resolution_mask against ground truth data."""
        # Test against ground truth using TorchTesting
        # Note: get_resolution_mask should have exact match for masks
        self.assertTrue(
            self.torch_testing.testTorchCallable(self.get_resolution_mask_log, get_resolution_mask),
            "get_resolution_mask failed ground truth test"
        )
        
        # Additional test for gradient flow through the resolution map
        cell = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], requires_grad=True)
        hkl_grid = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        _, res_map = get_resolution_mask(cell, hkl_grid, 1.0)
        loss = res_map.sum()
        loss.backward()
        
        self.assertIsNotNone(cell.grad)
        self.assertFalse(torch.allclose(cell.grad, torch.zeros_like(cell.grad)))

if __name__ == '__main__':
    unittest.main()
