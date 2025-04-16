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
        
        # --- Add comparison for raw eigenvectors (v_all) ---
        print("\nComparing raw eigenvectors (v_all) from eigh...")

        # Need to get v_all from np_model (requires modification or re-computation)
        # Recompute Dmat for NumPy for a specific index (e.g., idx=1)
        np_hessian = np_model.compute_hessian()
        np_kvec_idx1 = np_model.kvec.reshape(total_k_points, 3)[1] # Get kvec for idx=1
        np_Kmat_idx1 = np_model.gnm.compute_K(np_hessian, kvec=np_kvec_idx1)
        np_Kmat_2d_idx1 = np_Kmat_idx1.reshape(dof_total, dof_total)
        np_Dmat_idx1 = np.matmul(np_model.Linv, np.matmul(np_Kmat_2d_idx1, np_model.Linv.T))
        # Make Dmat Hermitian to match PyTorch behavior
        np_Dmat_hermitian_idx1 = 0.5 * (np_Dmat_idx1 + np_Dmat_idx1.T.conj())
        _, np_v_all_idx1 = np.linalg.eigh(np_Dmat_hermitian_idx1) # Get NumPy raw eigenvectors

        # Get PyTorch raw eigenvectors
        # Recompute Dmat for PyTorch for index 1
        torch_hessian = torch_model.compute_hessian()
        torch_kvec_idx1 = torch_model.kvec[1]
        # Need access to compute_K and Linv from torch_model
        from eryx.pdb_torch import GaussianNetworkModel as GNMTorch
        gnm_torch = GNMTorch()
        gnm_torch.n_asu = torch_model.n_asu
        gnm_torch.n_atoms_per_asu = torch_model.n_atoms_per_asu
        gnm_torch.n_cell = torch_model.n_cell
        gnm_torch.id_cell_ref = torch_model.id_cell_ref
        gnm_torch.device = torch_model.device
        gnm_torch.real_dtype = torch_model.real_dtype
        gnm_torch.complex_dtype = torch_model.complex_dtype
        gnm_torch.crystal = torch_model.crystal
        
        torch_Kmat_idx1 = gnm_torch.compute_K(torch_hessian, torch_kvec_idx1.unsqueeze(0))[0]
        torch_Kmat_2d_idx1 = torch_Kmat_idx1.reshape(dof_total, dof_total)
        torch_Linv_complex = torch_model.Linv.to(torch_model.complex_dtype)
        torch_Dmat_idx1 = torch.matmul(torch_Linv_complex, torch.matmul(torch_Kmat_2d_idx1, torch_Linv_complex.T))
        torch_Dmat_hermitian_idx1 = 0.5 * (torch_Dmat_idx1 + torch_Dmat_idx1.transpose(-2, -1).conj())
        _, torch_v_all_idx1 = torch.linalg.eigh(torch_Dmat_hermitian_idx1) # Get PyTorch raw eigenvectors

        torch_v_all_idx1_np = torch_v_all_idx1.detach().cpu().numpy()

        # Compare absolute values
        v_all_match = np.allclose(np.abs(np_v_all_idx1), np.abs(torch_v_all_idx1_np),
                                rtol=self.rtol, atol=self.atol)
        print(f"Raw eigenvectors (v_all) absolute values match for index 1: {v_all_match}")
        if not v_all_match:
            max_v_all_diff = np.max(np.abs(np.abs(np_v_all_idx1) - np.abs(torch_v_all_idx1_np)))
            print(f"Max abs diff in raw eigenvectors (v_all): {max_v_all_diff}")
        # --- End of added comparison ---

        # Compare projection matrices P = V @ V^H
        print("\nComparing projection matrices P = V @ V^H...")
        np_P = np.matmul(np_V, np.conjugate(np_V.transpose(0, 2, 1))) # V @ V^H for each k-point
        torch_P = np.matmul(torch_V_np, np.conjugate(torch_V_np.transpose(0, 2, 1)))

        V_match = np.allclose(np_P, torch_P, rtol=self.rtol, atol=self.atol, equal_nan=True) # Compare P matrices
        print(f"Projection matrices match: {V_match}")

        if not V_match:
            max_P_diff = np.max(np.abs(np_P - torch_P))
            print(f"Max projection matrix difference: {max_P_diff:.8e}")
            # Optionally print diff for a specific index
            idx=1
            print(f"Max P diff for idx {idx}: {np.max(np.abs(np_P[idx] - torch_P[idx])):.8e}")
        
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
        self.assertTrue(V_match, "Projection matrices (V @ V^H) do not match between NumPy and PyTorch")
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

    # Removed test_bz_averaged_adp_equivalence as it compared an attribute
    # that is no longer calculated/stored in the same way after refactoring.
    # The core ADP calculation equivalence is checked in test_compute_covariance_matrix_equivalence.

if __name__ == '__main__':
    unittest.main()
