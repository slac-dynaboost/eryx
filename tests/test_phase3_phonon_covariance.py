"""
Test suite for Phase 3: Verifying phonon and covariance matrix calculations.

This module tests that the PyTorch implementation of compute_gnm_phonons and
compute_covariance_matrix produces numerically identical results to their NumPy
counterparts in grid mode, using the 5zck_p1.pdb test file.
"""

import unittest
import numpy as np
import torch
import os
from typing import Dict, Any, Tuple, Optional

from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.models_torch import OnePhonon as TorchOnePhonon
from tests.torch_test_utils import TensorComparison


class TestPhononAndCovarianceEquivalence(unittest.TestCase):
    """
    Test suite for verifying numerical equivalence between NumPy and PyTorch
    implementations of phonon and covariance matrix calculations.
    """
    
    def setUp(self):
        """Set up test parameters and paths."""
        # Define standard parameters
        self.pdb_path = 'tests/pdbs/5zck_p1.pdb'
        
        # Use small sampling for faster tests
        self.hsampling = [-2, 2, 2]
        self.ksampling = [-2, 2, 2]
        self.lsampling = [-2, 2, 2]
        
        # Set device to CPU for deterministic results
        self.device = torch.device('cpu')
        
        # Define tolerances (slightly looser for phonon calculations)
        self.rtol = 1e-5
        self.atol = 1e-7
        
        # Ensure test PDB exists
        if not os.path.exists(self.pdb_path):
            raise FileNotFoundError(f"Test PDB file not found: {self.pdb_path}")
    
    def _create_models(self, params: Optional[Dict[str, Any]] = None) -> Tuple[NumpyOnePhonon, TorchOnePhonon]:
        """
        Create NumPy and PyTorch model instances with the same parameters.
        
        Args:
            params: Optional dictionary of parameters to override defaults
            
        Returns:
            Tuple of (numpy_model, torch_model)
        """
        # Start with default parameters
        model_params = {
            'pdb_path': self.pdb_path,
            'hsampling': self.hsampling,
            'ksampling': self.ksampling,
            'lsampling': self.lsampling,
            'expand_p1': True,
            'group_by': 'asu',
            'res_limit': 0.0,
            'model': 'gnm',
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0,
            'n_processes': 1  # Use single process for deterministic results
        }
        
        # Override with any provided parameters
        if params:
            model_params.update(params)
        
        # Create NumPy model
        np_model = NumpyOnePhonon(**model_params)
        
        # Create PyTorch model with device specification
        torch_params = model_params.copy()
        torch_params['device'] = self.device
        torch_model = TorchOnePhonon(**torch_params)
        
        return np_model, torch_model
    
    def test_compute_gnm_phonons_equivalence(self):
        """
        Test that compute_gnm_phonons produces equivalent results in NumPy and PyTorch.
        
        This test verifies that the eigendecomposition results (V and Winv) match
        between the NumPy and PyTorch implementations.
        """
        # Create models
        np_model, torch_model = self._create_models()
        
        # Explicitly call compute_gnm_phonons on both models
        np_model.compute_gnm_phonons()
        torch_model.compute_gnm_phonons()
        
        # Determine BZ dimensions from sampling rates
        h_dim_bz = int(self.hsampling[2])
        k_dim_bz = int(self.ksampling[2])
        l_dim_bz = int(self.lsampling[2])
        total_k_points = h_dim_bz * k_dim_bz * l_dim_bz
        
        # Get dimensions for reshaping
        n_asu = np_model.n_asu
        n_dof_per_asu = np_model.n_dof_per_asu
        dof_total = n_asu * n_dof_per_asu
        
        # Reshape NumPy V and Winv to match PyTorch shapes
        np_V = np_model.V.reshape(total_k_points, dof_total, dof_total)
        np_Winv = np_model.Winv.reshape(total_k_points, dof_total)
        
        # Convert PyTorch tensors to NumPy arrays
        torch_V_np = torch_model.V.detach().cpu().numpy()
        torch_Winv_np = torch_model.Winv.detach().cpu().numpy()
        
        # Compare absolute values of V (eigenvectors) since signs can differ
        V_match = np.allclose(np.abs(np_V), np.abs(torch_V_np), 
                             rtol=self.rtol, atol=self.atol, equal_nan=True)
        
        # Compare Winv (eigenvalues)
        Winv_match = np.allclose(np_Winv, torch_Winv_np, 
                                rtol=self.rtol, atol=self.atol, equal_nan=True)
        
        # Print detailed comparison for debugging
        if not V_match or not Winv_match:
            # Select a specific point for detailed comparison
            idx = 1  # Corresponds to dh=0, dk=0, dl=1 for 2x2x2 BZ
            print(f"\nDetailed comparison for BZ index {idx}:")
            print(f"NumPy V[{idx},0,0] (abs): {np.abs(np_V[idx,0,0]):.8e}")
            print(f"Torch V[{idx},0,0] (abs): {np.abs(torch_V_np[idx,0,0]):.8e}")
            print(f"NumPy Winv[{idx},0]: {np_Winv[idx,0]:.8e}")
            print(f"Torch Winv[{idx},0]: {torch_Winv_np[idx,0]:.8e}")
            
            # Calculate differences
            V_diff = np.abs(np_V) - np.abs(torch_V_np)
            Winv_diff = np_Winv - torch_Winv_np
            
            print(f"Max V diff: {np.nanmax(np.abs(V_diff)):.8e}")
            print(f"Max Winv diff: {np.nanmax(np.abs(Winv_diff)):.8e}")
        
        # Assert that both match
        self.assertTrue(V_match, "Eigenvectors (V) do not match between NumPy and PyTorch")
        self.assertTrue(Winv_match, "Eigenvalues (Winv) do not match between NumPy and PyTorch")
    
    def test_compute_covariance_matrix_equivalence(self):
        """
        Test that compute_covariance_matrix produces equivalent results in NumPy and PyTorch.
        
        This test verifies that the covariance matrix (covar) and atomic displacement
        parameters (ADP) match between the NumPy and PyTorch implementations.
        """
        # Create models
        np_model, torch_model = self._create_models()
        
        # Ensure phonons have been computed
        np_model.compute_gnm_phonons()
        torch_model.compute_gnm_phonons()
        
        # Explicitly call compute_covariance_matrix on both models
        np_model.compute_covariance_matrix()
        torch_model.compute_covariance_matrix()
        
        # Convert PyTorch tensors to NumPy arrays
        torch_covar_np = torch_model.covar.detach().cpu().numpy()
        torch_ADP_np = torch_model.ADP.detach().cpu().numpy()
        
        # Compare covariance matrices
        covar_match = np.allclose(np_model.covar, torch_covar_np, 
                                 rtol=self.rtol, atol=self.atol, equal_nan=True)
        
        # Compare ADPs
        ADP_match = np.allclose(np_model.ADP, torch_ADP_np, 
                               rtol=self.rtol, atol=self.atol, equal_nan=True)
        
        # Print detailed comparison for debugging
        if not covar_match or not ADP_match:
            print("\nDetailed covariance comparison:")
            print(f"NumPy covar shape: {np_model.covar.shape}")
            print(f"Torch covar shape: {torch_covar_np.shape}")
            
            # Compare first few elements
            print(f"NumPy covar[0,0,0,0,0]: {np_model.covar[0,0,0,0,0]:.8e}")
            print(f"Torch covar[0,0,0,0,0]: {torch_covar_np[0,0,0,0,0]:.8e}")
            
            # Calculate differences
            covar_diff = np_model.covar - torch_covar_np
            ADP_diff = np_model.ADP - torch_ADP_np
            
            print(f"Max covar diff: {np.nanmax(np.abs(covar_diff)):.8e}")
            print(f"Max ADP diff: {np.nanmax(np.abs(ADP_diff)):.8e}")
            
            # Print ADP comparison
            print("\nADP comparison:")
            print(f"NumPy ADP shape: {np_model.ADP.shape}")
            print(f"Torch ADP shape: {torch_ADP_np.shape}")
            print(f"NumPy ADP[0]: {np_model.ADP[0]:.8e}")
            print(f"Torch ADP[0]: {torch_ADP_np[0]:.8e}")
        
        # Assert that both match
        self.assertTrue(covar_match, "Covariance matrices do not match between NumPy and PyTorch")
        self.assertTrue(ADP_match, "ADPs do not match between NumPy and PyTorch")


if __name__ == '__main__':
    unittest.main()
