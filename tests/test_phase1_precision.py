"""
Test script to verify Phase 1 precision improvements.

This script compares the NumPy and PyTorch implementations of the OnePhonon model
to ensure that the foundational grid calculations and precision match exactly.
"""

import os
import sys
import numpy as np
import torch
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.models_torch import OnePhonon as TorchOnePhonon

class TestPhase1Precision(unittest.TestCase):
    """Test case for verifying Phase 1 precision improvements."""
    
    def setUp(self):
        """Set up test parameters."""
        # Define common parameters for both models
        # Try to find a suitable PDB file from the available ones in the repository
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'pdbs', '193l.pdb'),
            os.path.join(os.path.dirname(__file__), 'pdbs', '2ol9.pdb'),
            os.path.join(os.path.dirname(__file__), 'pdbs', '5zck.pdb'),
            os.path.join(os.path.dirname(__file__), 'pdbs', '7n2h.pdb'),
            os.path.join(os.path.dirname(__file__), 'pdbs', 'histidine.pdb')
        ]
        
        self.pdb_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.pdb_path = path
                break
                
        if self.pdb_path is None:
            raise FileNotFoundError(f"Could not find any suitable PDB file for testing")
        
        # Define sampling parameters
        self.hsampling = [-4, 4, 3]
        self.ksampling = [-4, 4, 3]
        self.lsampling = [-4, 4, 3]
        
        # Other common parameters
        self.expand_p1 = True
        self.group_by = 'asu'
        self.res_limit = 0.0
        self.model = 'gnm'
        self.gnm_cutoff = 4.0
        self.gamma_intra = 1.0
        self.gamma_inter = 1.0
        
        # Tolerance for numerical comparisons
        self.rtol = 1e-14  # Relative tolerance
        self.atol = 1e-14  # Absolute tolerance
        
        print(f"Using test PDB file: {self.pdb_path}")
    
    def test_grid_precision(self):
        """Test that grid calculations match between NumPy and PyTorch implementations."""
        print("\n=== Testing Grid Precision ===")
        
        # Initialize NumPy model
        print("Initializing NumPy model...")
        np_model = NumpyOnePhonon(
            self.pdb_path,
            hsampling=self.hsampling,
            ksampling=self.ksampling,
            lsampling=self.lsampling,
            expand_p1=self.expand_p1,
            group_by=self.group_by,
            res_limit=self.res_limit,
            model=self.model,
            gnm_cutoff=self.gnm_cutoff,
            gamma_intra=self.gamma_intra,
            gamma_inter=self.gamma_inter
        )
        
        # Initialize PyTorch model
        print("Initializing PyTorch model...")
        torch_model = TorchOnePhonon(
            self.pdb_path,
            hsampling=self.hsampling,
            ksampling=self.ksampling,
            lsampling=self.lsampling,
            expand_p1=self.expand_p1,
            group_by=self.group_by,
            res_limit=self.res_limit,
            model=self.model,
            gnm_cutoff=self.gnm_cutoff,
            gamma_intra=self.gamma_intra,
            gamma_inter=self.gamma_inter
        )
        
        # Compare map_shape
        print(f"NumPy map_shape: {np_model.map_shape}")
        print(f"PyTorch map_shape: {torch_model.map_shape}")
        self.assertEqual(np_model.map_shape, torch_model.map_shape, 
                         "map_shape does not match between NumPy and PyTorch models")
        
        # Compare hkl_grid
        print("Comparing hkl_grid...")
        np_hkl_grid = np_model.hkl_grid
        torch_hkl_grid = torch_model.hkl_grid.detach().cpu().numpy()
        
        print(f"NumPy hkl_grid shape: {np_hkl_grid.shape}, dtype: {np_hkl_grid.dtype}")
        print(f"PyTorch hkl_grid shape: {torch_hkl_grid.shape}, dtype: {torch_hkl_grid.dtype}")
        
        # Print first few values for visual inspection
        print(f"NumPy hkl_grid[0:3]: {np_hkl_grid[0:3]}")
        print(f"PyTorch hkl_grid[0:3]: {torch_hkl_grid[0:3]}")
        
        # Check if shapes match
        self.assertEqual(np_hkl_grid.shape, torch_hkl_grid.shape,
                         "hkl_grid shapes do not match")
        
        # Check if values match within tolerance
        is_close = np.allclose(np_hkl_grid, torch_hkl_grid, rtol=self.rtol, atol=self.atol)
        if not is_close:
            # Find max difference for debugging
            max_diff = np.max(np.abs(np_hkl_grid - torch_hkl_grid))
            print(f"Max difference in hkl_grid: {max_diff}")
            
            # Find indices of maximum difference
            max_idx = np.unravel_index(np.argmax(np.abs(np_hkl_grid - torch_hkl_grid)), np_hkl_grid.shape)
            print(f"Max difference at index {max_idx}:")
            print(f"NumPy value: {np_hkl_grid[max_idx]}")
            print(f"PyTorch value: {torch_hkl_grid[max_idx]}")
        
        self.assertTrue(is_close, "hkl_grid values do not match within tolerance")
        print("✓ hkl_grid matches between NumPy and PyTorch models")
        
        # Compare q_grid
        print("Comparing q_grid...")
        np_q_grid = np_model.q_grid
        torch_q_grid = torch_model.q_grid.detach().cpu().numpy()
        
        print(f"NumPy q_grid shape: {np_q_grid.shape}, dtype: {np_q_grid.dtype}")
        print(f"PyTorch q_grid shape: {torch_q_grid.shape}, dtype: {torch_q_grid.dtype}")
        
        # Print first few values for visual inspection
        print(f"NumPy q_grid[0:3]: {np_q_grid[0:3]}")
        print(f"PyTorch q_grid[0:3]: {torch_q_grid[0:3]}")
        
        # Check if shapes match
        self.assertEqual(np_q_grid.shape, torch_q_grid.shape,
                         "q_grid shapes do not match")
        
        # Check if values match within tolerance
        is_close = np.allclose(np_q_grid, torch_q_grid, rtol=self.rtol, atol=self.atol)
        if not is_close:
            # Find max difference for debugging
            max_diff = np.max(np.abs(np_q_grid - torch_q_grid))
            print(f"Max difference in q_grid: {max_diff}")
            
            # Find indices of maximum difference
            max_idx = np.unravel_index(np.argmax(np.abs(np_q_grid - torch_q_grid)), np_q_grid.shape)
            print(f"Max difference at index {max_idx}:")
            print(f"NumPy value: {np_q_grid[max_idx]}")
            print(f"PyTorch value: {torch_q_grid[max_idx]}")
        
        self.assertTrue(is_close, "q_grid values do not match within tolerance")
        print("✓ q_grid matches between NumPy and PyTorch models")
        
        # Compare kvec
        print("Comparing kvec...")
        # Reshape NumPy kvec to match PyTorch kvec shape
        h_dim, k_dim, l_dim = np_model.map_shape
        np_kvec_reshaped = np_model.kvec.reshape(h_dim * k_dim * l_dim, 3)
        torch_kvec = torch_model.kvec.detach().cpu().numpy()
        
        print(f"NumPy kvec shape (original): {np_model.kvec.shape}")
        print(f"NumPy kvec shape (reshaped): {np_kvec_reshaped.shape}")
        print(f"PyTorch kvec shape: {torch_kvec.shape}")
        
        # Print first few values for visual inspection
        print(f"NumPy kvec[0:3]: {np_kvec_reshaped[0:3]}")
        print(f"PyTorch kvec[0:3]: {torch_kvec[0:3]}")
        
        # Check if shapes match
        self.assertEqual(np_kvec_reshaped.shape, torch_kvec.shape,
                         "kvec shapes do not match after reshaping")
        
        # Check if values match within tolerance
        is_close = np.allclose(np_kvec_reshaped, torch_kvec, rtol=self.rtol, atol=self.atol)
        if not is_close:
            # Find max difference for debugging
            max_diff = np.max(np.abs(np_kvec_reshaped - torch_kvec))
            print(f"Max difference in kvec: {max_diff}")
            
            # Find indices of maximum difference
            max_idx = np.unravel_index(np.argmax(np.abs(np_kvec_reshaped - torch_kvec)), np_kvec_reshaped.shape)
            print(f"Max difference at index {max_idx}:")
            print(f"NumPy value: {np_kvec_reshaped[max_idx]}")
            print(f"PyTorch value: {torch_kvec[max_idx]}")
        
        self.assertTrue(is_close, "kvec values do not match within tolerance")
        print("✓ kvec matches between NumPy and PyTorch models")
        
        # Compare kvec_norm
        print("Comparing kvec_norm...")
        # Reshape NumPy kvec_norm to match PyTorch kvec_norm shape
        np_kvec_norm_reshaped = np_model.kvec_norm.reshape(h_dim * k_dim * l_dim, 1)
        torch_kvec_norm = torch_model.kvec_norm.detach().cpu().numpy()
        
        print(f"NumPy kvec_norm shape (original): {np_model.kvec_norm.shape}")
        print(f"NumPy kvec_norm shape (reshaped): {np_kvec_norm_reshaped.shape}")
        print(f"PyTorch kvec_norm shape: {torch_kvec_norm.shape}")
        
        # Print first few values for visual inspection
        print(f"NumPy kvec_norm[0:3]: {np_kvec_norm_reshaped[0:3]}")
        print(f"PyTorch kvec_norm[0:3]: {torch_kvec_norm[0:3]}")
        
        # Check if shapes match
        self.assertEqual(np_kvec_norm_reshaped.shape, torch_kvec_norm.shape,
                         "kvec_norm shapes do not match after reshaping")
        
        # Check if values match within tolerance
        is_close = np.allclose(np_kvec_norm_reshaped, torch_kvec_norm, rtol=self.rtol, atol=self.atol)
        if not is_close:
            # Find max difference for debugging
            max_diff = np.max(np.abs(np_kvec_norm_reshaped - torch_kvec_norm))
            print(f"Max difference in kvec_norm: {max_diff}")
            
            # Find indices of maximum difference
            max_idx = np.unravel_index(np.argmax(np.abs(np_kvec_norm_reshaped - torch_kvec_norm)), np_kvec_norm_reshaped.shape)
            print(f"Max difference at index {max_idx}:")
            print(f"NumPy value: {np_kvec_norm_reshaped[max_idx]}")
            print(f"PyTorch value: {torch_kvec_norm[max_idx]}")
        
        self.assertTrue(is_close, "kvec_norm values do not match within tolerance")
        print("✓ kvec_norm matches between NumPy and PyTorch models")
        
        print("\nAll grid precision tests passed! Phase 1 is complete.")

if __name__ == '__main__':
    unittest.main()
