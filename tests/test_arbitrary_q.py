"""
Tests for arbitrary q-vector input support in the OnePhonon model.

This module contains tests to verify the functionality of arbitrary q-vector
inputs in the PyTorch implementation of diffuse scattering calculations.
"""

import os
import unittest
import torch
import numpy as np
from typing import Optional, List, Tuple

from tests.test_base import TestBase
from eryx.models_torch import OnePhonon


class TestArbitraryQVectors(TestBase):
    """Test case for arbitrary q-vector input support."""
    
    def setUp(self):
        """Set up test environment."""
        # Call parent setUp
        super().setUp()
        
        # Set module name for log paths
        self.module_name = "eryx.models_torch"
        self.class_name = "OnePhonon"
        
        # Default test parameters
        self.pdb_path = 'tests/pdbs/5zck_p1.pdb'
        
        # Grid parameters for comparison
        self.grid_params = {
            'hsampling': [-2, 2, 3],
            'ksampling': [-2, 2, 3],
            'lsampling': [-2, 2, 3],
            'expand_p1': True,
            'res_limit': 0.0
        }
    
    def create_q_vectors_from_grid(self) -> Tuple[torch.Tensor, OnePhonon]:
        """
        Create q-vectors from grid-based approach for equivalence testing.
        
        Returns:
            Tuple containing:
                - q_vectors: Tensor of q-vectors from grid
                - grid_model: OnePhonon instance using grid-based approach
        """
        # Create grid-based model
        grid_model = OnePhonon(
            self.pdb_path,
            **self.grid_params,
            device=self.device
        )
        
        # Extract q-vectors from grid model
        q_vectors = grid_model.q_grid.clone().detach()
        
        return q_vectors, grid_model
    
    def test_constructor_validation(self):
        """Test constructor validation for q-vectors parameter."""
        # Test with invalid q-vectors type
        with self.assertRaises(ValueError):
            OnePhonon(
                self.pdb_path,
                q_vectors=np.array([[0.1, 0.2, 0.3]]),  # NumPy array instead of tensor
                device=self.device
            )
        
        # Test with invalid q-vectors shape
        with self.assertRaises(ValueError):
            OnePhonon(
                self.pdb_path,
                q_vectors=torch.tensor([0.1, 0.2, 0.3]),  # 1D tensor instead of 2D
                device=self.device
            )
        
        # Test with missing required parameters
        with self.assertRaises(ValueError):
            OnePhonon(
                self.pdb_path,
                hsampling=None,  # Missing required parameter
                ksampling=[-2, 2, 3],
                lsampling=[-2, 2, 3],
                device=self.device
            )
        
        # Test with valid q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=torch.tensor([[0.1, 0.2, 0.3]], device=self.device),
            device=self.device
        )
        
        # Verify model attributes
        self.assertTrue(model.use_arbitrary_q)
        self.assertEqual(model.q_grid.shape, (1, 3))
        self.assertTrue(model.q_grid.requires_grad)
    
    def test_grid_equivalence(self):
        """
        Test that using explicit q-vectors from a grid produces identical results
        to the grid-based approach.
        """
        # Get q-vectors from grid model
        q_vectors, grid_model = self.create_q_vectors_from_grid()
        
        # Create arbitrary q-vector model with the same q-vectors
        q_model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Compare q_grid tensors (should be identical)
        self.assertTrue(torch.allclose(grid_model.q_grid, q_model.q_grid))
        
        # Compare hkl_grid tensors (should be close within numerical precision)
        self.assertTrue(torch.allclose(grid_model.hkl_grid, q_model.hkl_grid, rtol=1e-5, atol=1e-8))
        
        # Compare resolution masks
        self.assertTrue(torch.all(grid_model.res_mask == q_model.res_mask))
    
    def test_gradient_flow(self):
        """
        Test that gradients flow correctly through arbitrary q-vector calculations.
        """
        # Create a small set of q-vectors that requires gradients
        q_vectors = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], device=self.device, requires_grad=True)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Verify q_grid has gradients enabled
        self.assertTrue(model.q_grid.requires_grad)
        
        # Create a simple computation to test gradient flow
        # Sum of all elements in q_grid
        q_sum = torch.sum(model.q_grid)
        
        # Compute backward pass
        q_sum.backward()
        
        # Verify gradients are computed
        self.assertIsNotNone(q_vectors.grad)
        self.assertTrue(torch.all(q_vectors.grad == torch.ones_like(q_vectors)))
    
    def test_custom_q_vectors(self):
        """
        Test with custom q-vectors that don't follow a grid pattern.
        """
        # Create a custom set of q-vectors
        q_vectors = torch.tensor([
            [0.123, 0.456, 0.789],
            [1.234, 2.345, 3.456],
            [-0.123, -0.456, -0.789]
        ], device=self.device)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Verify model attributes
        self.assertTrue(model.use_arbitrary_q)
        self.assertEqual(model.q_grid.shape, (3, 3))
        self.assertTrue(torch.allclose(model.q_grid, q_vectors))
        
        # Verify map_shape is set correctly
        self.assertEqual(model.map_shape, (3, 1, 1))
        
        # Verify hkl_grid is computed correctly
        # q = 2π * A_inv^T * hkl, so hkl = (1/2π) * q * (A_inv^T)^-1
        A_inv_tensor = torch.tensor(model.model.A_inv, dtype=torch.float32, device=self.device)
        scaling_factor = 1.0 / (2.0 * torch.pi)
        A_inv_T_inv = torch.inverse(A_inv_tensor.T)
        expected_hkl = torch.matmul(q_vectors * scaling_factor, A_inv_T_inv)
        
        self.assertTrue(torch.allclose(model.hkl_grid, expected_hkl, rtol=1e-5, atol=1e-8))


if __name__ == '__main__':
    unittest.main()
