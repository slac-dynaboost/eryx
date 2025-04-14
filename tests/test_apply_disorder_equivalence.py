"""
Test equivalence between NumPy and PyTorch implementations of apply_disorder.

This module verifies that the PyTorch implementation of apply_disorder in
eryx.models_torch.OnePhonon produces numerically equivalent results to the
NumPy implementation in eryx.models.OnePhonon when operating in grid-based mode.
"""

import unittest
import numpy as np
import torch
import os
import tempfile

from eryx.models import OnePhonon as OnePhononNumPy
from eryx.models_torch import OnePhonon as OnePhononTorch


class TestApplyDisorderEquivalence(unittest.TestCase):
    """
    Test equivalence between NumPy and PyTorch implementations of apply_disorder.
    
    This test class verifies that the PyTorch implementation of apply_disorder
    produces numerically equivalent results to the NumPy implementation when
    operating in grid-based mode.
    """
    
    def setUp(self):
        """Set up test parameters and models."""
        # Use a small PDB file for testing
        self.pdb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdbs", "5zck.pdb")
        
        # Use minimal sampling parameters for faster testing
        self.hsampling = [-1, 1, 2]  # min, max, steps
        self.ksampling = [-1, 1, 2]
        self.lsampling = [-1, 1, 2]
        
        # Other common parameters
        self.expand_p1 = True
        self.group_by = 'asu'
        self.res_limit = 0.0
        self.model = 'gnm'
        self.gnm_cutoff = 4.0
        self.gamma_intra = 1.0
        self.gamma_inter = 1.0
        self.n_processes = 1
        
        # Set device for PyTorch model
        self.device = torch.device('cpu')
        
        # Create NumPy model
        self.np_model = OnePhononNumPy(
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
            gamma_inter=self.gamma_inter,
            n_processes=self.n_processes
        )
        
        # Create PyTorch model
        self.torch_model = OnePhononTorch(
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
            gamma_inter=self.gamma_inter,
            n_processes=self.n_processes,
            device=self.device
        )
        
        # Run prerequisite calculations
        print("Computing NumPy phonons...")
        self.np_model.compute_gnm_phonons()
        self.np_model.compute_covariance_matrix()
        
        print("Computing PyTorch phonons...")
        self.torch_model.compute_gnm_phonons()
        self.torch_model.compute_covariance_matrix()
    
    def test_equivalence_all_modes(self):
        """Test equivalence of apply_disorder with all modes (rank=-1) and data ADPs."""
        print("\nTesting apply_disorder equivalence with all modes (rank=-1) and data ADPs...")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run both implementations
            Id_np = self.np_model.apply_disorder(rank=-1, use_data_adp=True, outdir=tmpdir)
            Id_torch = self.torch_model.apply_disorder(rank=-1, use_data_adp=True, outdir=tmpdir)
            
            # Convert PyTorch tensor to NumPy array
            Id_torch_np = Id_torch.detach().cpu().numpy()
            
            # Compare arrays
            # Use equal_nan=True to handle NaN values (outside resolution mask)
            is_close = np.allclose(Id_np, Id_torch_np, rtol=1e-5, atol=1e-7, equal_nan=True)
            
            # Print detailed comparison if test fails
            if not is_close:
                # Calculate differences
                abs_diff = np.abs(Id_np - Id_torch_np)
                rel_diff = abs_diff / (np.abs(Id_np) + 1e-10)
                
                # Filter out NaN values
                valid_mask = ~np.isnan(Id_np) & ~np.isnan(Id_torch_np)
                
                if np.any(valid_mask):
                    max_abs_diff = np.max(abs_diff[valid_mask])
                    max_rel_diff = np.max(rel_diff[valid_mask])
                    max_abs_idx = np.unravel_index(np.argmax(abs_diff * valid_mask), abs_diff.shape)
                    
                    print(f"Max absolute difference: {max_abs_diff} at index {max_abs_idx}")
                    print(f"Max relative difference: {max_rel_diff}")
                    print(f"NumPy value at max diff: {Id_np[max_abs_idx]}")
                    print(f"PyTorch value at max diff: {Id_torch_np[max_abs_idx]}")
                else:
                    print("No valid (non-NaN) values to compare")
            
            self.assertTrue(is_close, "NumPy and PyTorch implementations produce different results")
    
    def test_equivalence_single_mode(self):
        """Test equivalence of apply_disorder with a single mode (rank=0) and data ADPs."""
        print("\nTesting apply_disorder equivalence with single mode (rank=0) and data ADPs...")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run both implementations
            Id_np = self.np_model.apply_disorder(rank=0, use_data_adp=True, outdir=tmpdir)
            Id_torch = self.torch_model.apply_disorder(rank=0, use_data_adp=True, outdir=tmpdir)
            
            # Convert PyTorch tensor to NumPy array
            Id_torch_np = Id_torch.detach().cpu().numpy()
            
            # Compare arrays
            is_close = np.allclose(Id_np, Id_torch_np, rtol=1e-5, atol=1e-7, equal_nan=True)
            
            # Print detailed comparison if test fails
            if not is_close:
                # Calculate differences
                abs_diff = np.abs(Id_np - Id_torch_np)
                rel_diff = abs_diff / (np.abs(Id_np) + 1e-10)
                
                # Filter out NaN values
                valid_mask = ~np.isnan(Id_np) & ~np.isnan(Id_torch_np)
                
                if np.any(valid_mask):
                    max_abs_diff = np.max(abs_diff[valid_mask])
                    max_rel_diff = np.max(rel_diff[valid_mask])
                    max_abs_idx = np.unravel_index(np.argmax(abs_diff * valid_mask), abs_diff.shape)
                    
                    print(f"Max absolute difference: {max_abs_diff} at index {max_abs_idx}")
                    print(f"Max relative difference: {max_rel_diff}")
                    print(f"NumPy value at max diff: {Id_np[max_abs_idx]}")
                    print(f"PyTorch value at max diff: {Id_torch_np[max_abs_idx]}")
                else:
                    print("No valid (non-NaN) values to compare")
            
            self.assertTrue(is_close, "NumPy and PyTorch implementations produce different results")
    
    def test_equivalence_computed_adp(self):
        """Test equivalence of apply_disorder with all modes and computed ADPs."""
        print("\nTesting apply_disorder equivalence with all modes and computed ADPs...")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run both implementations
            Id_np = self.np_model.apply_disorder(rank=-1, use_data_adp=False, outdir=tmpdir)
            Id_torch = self.torch_model.apply_disorder(rank=-1, use_data_adp=False, outdir=tmpdir)
            
            # Convert PyTorch tensor to NumPy array
            Id_torch_np = Id_torch.detach().cpu().numpy()
            
            # Compare arrays
            is_close = np.allclose(Id_np, Id_torch_np, rtol=1e-5, atol=1e-7, equal_nan=True)
            
            # Print detailed comparison if test fails
            if not is_close:
                # Calculate differences
                abs_diff = np.abs(Id_np - Id_torch_np)
                rel_diff = abs_diff / (np.abs(Id_np) + 1e-10)
                
                # Filter out NaN values
                valid_mask = ~np.isnan(Id_np) & ~np.isnan(Id_torch_np)
                
                if np.any(valid_mask):
                    max_abs_diff = np.max(abs_diff[valid_mask])
                    max_rel_diff = np.max(rel_diff[valid_mask])
                    max_abs_idx = np.unravel_index(np.argmax(abs_diff * valid_mask), abs_diff.shape)
                    
                    print(f"Max absolute difference: {max_abs_diff} at index {max_abs_idx}")
                    print(f"Max relative difference: {max_rel_diff}")
                    print(f"NumPy value at max diff: {Id_np[max_abs_idx]}")
                    print(f"PyTorch value at max diff: {Id_torch_np[max_abs_idx]}")
                else:
                    print("No valid (non-NaN) values to compare")
            
            self.assertTrue(is_close, "NumPy and PyTorch implementations produce different results")


if __name__ == "__main__":
    unittest.main()
