import unittest
import os
import torch
import numpy as np
from eryx.scatter_torch import compute_form_factors, structure_factors_batch, structure_factors
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping

class TestScatterTorch(unittest.TestCase):
    def setUp(self):
        # Set up the testing framework
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        # Use tolerances specified in the documentation
        self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-4, atol=1e-7)
        
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
        
        # Log file prefixes for ground truth data
        self.compute_form_factors_log = "logs/eryx.scatter.compute_form_factors"
        self.structure_factors_batch_log = "logs/eryx.scatter.structure_factors_batch"
        self.structure_factors_log = "logs/eryx.scatter.structure_factors"
        
        # Ensure log files exist
        for log_file in [self.compute_form_factors_log, self.structure_factors_batch_log, self.structure_factors_log]:
            log_file_path = f"{log_file}.log"
            self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
    
    def test_compute_form_factors(self):
        """Test compute_form_factors against ground truth data."""
        # Test against ground truth using TorchTesting
        self.assertTrue(
            self.torch_testing.testTorchCallable(self.compute_form_factors_log, compute_form_factors),
            "compute_form_factors failed ground truth test"
        )
        
        # Additional test for gradient flow
        q_grid = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3]
        ], requires_grad=True, dtype=torch.float32)
        
        ff_a = torch.tensor([
            [2.31000, 1.02000, 1.58860, 0.86500],
            [2.31000, 1.02000, 1.58860, 0.86500]
        ], dtype=torch.float32)
        
        ff_b = torch.tensor([
            [20.8439, 10.2075, 0.5687, 51.6512],
            [20.8439, 10.2075, 0.5687, 51.6512]
        ], dtype=torch.float32)
        
        ff_c = torch.tensor([0.2159, 0.2159], dtype=torch.float32)
        
        form_factors = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
        loss = form_factors.sum()
        loss.backward()
        
        self.assertIsNotNone(q_grid.grad)
        self.assertFalse(torch.allclose(q_grid.grad, torch.zeros_like(q_grid.grad)))
    
    def test_structure_factors_batch(self):
        """Test structure_factors_batch against ground truth data."""
        # Test against ground truth using TorchTesting
        self.assertTrue(
            self.torch_testing.testTorchCallable(self.structure_factors_batch_log, structure_factors_batch),
            "structure_factors_batch failed ground truth test"
        )
        
        # Additional test for gradient flow
        q_grid = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3]
        ], requires_grad=True, dtype=torch.float32)
        
        xyz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        ff_a = torch.tensor([
            [2.31000, 1.02000, 1.58860, 0.86500],
            [2.31000, 1.02000, 1.58860, 0.86500]
        ], dtype=torch.float32)
        
        ff_b = torch.tensor([
            [20.8439, 10.2075, 0.5687, 51.6512],
            [20.8439, 10.2075, 0.5687, 51.6512]
        ], dtype=torch.float32)
        
        ff_c = torch.tensor([0.2159, 0.2159], dtype=torch.float32)
        
        U = torch.tensor([0.05, 0.05], dtype=torch.float32)
        
        sf = structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, U=U)
        loss = sf.abs().sum()
        loss.backward()
        
        self.assertIsNotNone(q_grid.grad)
        self.assertFalse(torch.allclose(q_grid.grad, torch.zeros_like(q_grid.grad)))

    def test_compute_qF_option(self):
        """Test the compute_qF option for structure factors calculation."""
        q_grid = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3]
        ], dtype=torch.float32)
        
        xyz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        ff_a = torch.tensor([
            [2.31000, 1.02000, 1.58860, 0.86500],
            [2.31000, 1.02000, 1.58860, 0.86500]
        ], dtype=torch.float32)
        
        ff_b = torch.tensor([
            [20.8439, 10.2075, 0.5687, 51.6512],
            [20.8439, 10.2075, 0.5687, 51.6512]
        ], dtype=torch.float32)
        
        ff_c = torch.tensor([0.2159, 0.2159], dtype=torch.float32)
        
        # With compute_qF=False
        sf_normal = structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, compute_qF=False, sum_over_atoms=True)
        
        # With compute_qF=True
        sf_qF = structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, compute_qF=True, sum_over_atoms=False)
        
        # Shape should be different
        self.assertEqual(sf_normal.shape[0], q_grid.shape[0])
        # When compute_qF=True and sum_over_atoms=False, shape should be (n_points, n_atoms*3)
        self.assertEqual(sf_qF.shape[0], q_grid.shape[0])
        self.assertEqual(sf_qF.shape[1], xyz.shape[0] * 3)

if __name__ == '__main__':
    unittest.main()
