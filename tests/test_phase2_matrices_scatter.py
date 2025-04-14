"""
Test suite for Phase 2: Verifying matrix calculations and scattering functions.

This test suite verifies that the core matrix calculations (Amat, M_allatoms, Mmat, Linv)
and the fundamental scattering calculation (structure_factors_batch) in the PyTorch
OnePhonon model produce results that are numerically identical (within specified tolerances)
to their counterparts in the original NumPy implementation when operating in grid mode.
"""

import unittest
import os
import numpy as np
import torch

# Import NumPy implementations
from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.scatter import structure_factors_batch as np_structure_factors_batch

# Import PyTorch implementations
from eryx.models_torch import OnePhonon as TorchOnePhonon
from eryx.scatter_torch import structure_factors_batch as torch_structure_factors_batch


class TestMatricesAndScatter(unittest.TestCase):
    """Test suite for verifying matrix calculations and scattering functions."""

    def setUp(self):
        """Set up test parameters and tolerances."""
        # Set up test parameters
        self.pdb_path = os.path.join(os.path.dirname(__file__), "pdbs", "5zck_p1.pdb")
        self.hsampling = [-4, 4, 3]
        self.ksampling = [-4, 4, 3]
        self.lsampling = [-4, 4, 3]
        self.expand_p1 = True
        self.group_by = 'asu'
        self.res_limit = 0.0
        self.model = 'gnm'
        self.gnm_cutoff = 4.0
        self.gamma_intra = 1.0
        self.gamma_inter = 1.0
        self.n_processes = 1
        
        # Set PyTorch device to CPU for deterministic comparisons
        self.device = torch.device('cpu')
        
        # Define tolerances for comparisons
        self.rtol_tight = 1e-12
        self.atol_tight = 1e-14
        self.rtol_looser = 1e-6
        self.atol_looser = 1e-8
        
        # Ensure the test PDB file exists
        if not os.path.exists(self.pdb_path):
            raise FileNotFoundError(f"Test PDB file not found: {self.pdb_path}")
    
    def _create_models(self, params=None):
        """
        Create NumPy and PyTorch model instances with identical parameters.
        
        Args:
            params: Optional dictionary of parameters to override defaults
            
        Returns:
            Tuple of (np_model, torch_model)
        """
        # Use default parameters if none provided
        if params is None:
            params = {
                'pdb_path': self.pdb_path,
                'hsampling': self.hsampling,
                'ksampling': self.ksampling,
                'lsampling': self.lsampling,
                'expand_p1': self.expand_p1,
                'group_by': self.group_by,
                'res_limit': self.res_limit,
                'model': self.model,
                'gnm_cutoff': self.gnm_cutoff,
                'gamma_intra': self.gamma_intra,
                'gamma_inter': self.gamma_inter,
                'n_processes': self.n_processes
            }
        
        # Create NumPy model
        np_model = NumpyOnePhonon(
            params['pdb_path'],
            params['hsampling'],
            params['ksampling'],
            params['lsampling'],
            params['expand_p1'],
            params['group_by'],
            params['res_limit'],
            params['model'],
            params['gnm_cutoff'],
            params['gamma_intra'],
            params['gamma_inter'],
            params['n_processes']
        )
        
        # Create PyTorch model with same parameters
        torch_model = TorchOnePhonon(
            params['pdb_path'],
            params['hsampling'],
            params['ksampling'],
            params['lsampling'],
            params['expand_p1'],
            params['group_by'],
            params['res_limit'],
            params['model'],
            params['gnm_cutoff'],
            params['gamma_intra'],
            params['gamma_inter'],
            params['n_processes'],
            device=self.device
        )
        
        return np_model, torch_model
    
    def test_Amat_equivalence(self):
        """Test that the projection matrix Amat is equivalent between NumPy and PyTorch implementations."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Access the projection matrices
        np_Amat = np_model.Amat
        torch_Amat = torch_model.Amat
        
        # Print shapes and dtypes for debugging
        print(f"NumPy Amat shape: {np_Amat.shape}, dtype: {np_Amat.dtype}")
        print(f"PyTorch Amat shape: {torch_Amat.shape}, dtype: {torch_Amat.dtype}")
        
        # Verify torch_Amat properties
        self.assertIsInstance(torch_Amat, torch.Tensor)
        self.assertEqual(torch_Amat.dtype, torch.float64)
        self.assertTrue(torch_Amat.requires_grad)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_Amat_np = torch_Amat.detach().cpu().numpy()
        
        # Compare the matrices
        self.assertTrue(
            np.allclose(np_Amat, torch_Amat_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Amat matrices are not equivalent within tolerance"
        )
    
    def test_M_allatoms_equivalence(self):
        """Test that the all-atom mass matrix M_allatoms is equivalent between NumPy and PyTorch implementations."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Explicitly call the methods to build M_allatoms
        np_M0 = np_model._build_M_allatoms()
        torch_M0 = torch_model._build_M_allatoms()
        
        # Print shapes and dtypes for debugging
        print(f"NumPy M_allatoms shape: {np_M0.shape}, dtype: {np_M0.dtype}")
        print(f"PyTorch M_allatoms shape: {torch_M0.shape}, dtype: {torch_M0.dtype}")
        
        # Verify torch_M0 properties
        self.assertIsInstance(torch_M0, torch.Tensor)
        self.assertEqual(torch_M0.dtype, torch.float64)
        self.assertTrue(torch_M0.requires_grad)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_M0_np = torch_M0.detach().cpu().numpy()
        
        # Compare the matrices
        self.assertTrue(
            np.allclose(np_M0, torch_M0_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "M_allatoms matrices are not equivalent within tolerance"
        )
    
    def test_project_M_equivalence(self):
        """Test that the projected mass matrix Mmat is equivalent between NumPy and PyTorch implementations."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Get the M_allatoms matrices
        np_M_allatoms = np_model._build_M_allatoms()
        torch_M_allatoms = torch_model._build_M_allatoms()
        
        # Verify M_allatoms matrices are numerically identical first
        torch_M_allatoms_np = torch_M_allatoms.detach().cpu().numpy()
        self.assertTrue(
            np.allclose(np_M_allatoms, torch_M_allatoms_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "M_allatoms matrices are not equivalent within tolerance"
        )
        
        # Get the Amat matrices
        np_Amat = np_model.Amat
        torch_Amat = torch_model.Amat
        
        # Verify Amat matrices are numerically identical first
        torch_Amat_np = torch_Amat.detach().cpu().numpy()
        self.assertTrue(
            np.allclose(np_Amat, torch_Amat_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Amat matrices are not equivalent within tolerance"
        )
        
        # Call the projection methods
        np_Mmat = np_model._project_M(np_M_allatoms)
        torch_Mmat = torch_model._project_M(torch_M_allatoms)
        
        # Print shapes and dtypes for debugging
        print(f"NumPy Mmat shape: {np_Mmat.shape}, dtype: {np_Mmat.dtype}")
        print(f"PyTorch Mmat shape: {torch_Mmat.shape}, dtype: {torch_Mmat.dtype}")
        
        # Verify torch_Mmat properties
        self.assertIsInstance(torch_Mmat, torch.Tensor)
        self.assertEqual(torch_Mmat.dtype, torch.float64)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_Mmat_np = torch_Mmat.detach().cpu().numpy()
        
        # Compare the matrices
        self.assertTrue(
            np.allclose(np_Mmat, torch_Mmat_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Projected Mmat matrices are not equivalent within tolerance"
        )
    
    def test_Linv_equivalence(self):
        """Test that the inverse Cholesky factor Linv is equivalent between NumPy and PyTorch implementations."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Access the resulting inverse factors
        np_Linv = np_model.Linv
        torch_Linv = torch_model.Linv
        
        # Print shapes and dtypes for debugging
        print(f"NumPy Linv shape: {np_Linv.shape}, dtype: {np_Linv.dtype}")
        print(f"PyTorch Linv shape: {torch_Linv.shape}, dtype: {torch_Linv.dtype}")
        
        # Verify torch_Linv properties
        self.assertIsInstance(torch_Linv, torch.Tensor)
        self.assertTrue(torch_Linv.requires_grad)
        
        # Check dtype - could be float64 or complex128 depending on the code path
        self.assertTrue(
            torch_Linv.dtype in [torch.float64, torch.complex128],
            f"Expected torch_Linv dtype to be float64 or complex128, got {torch_Linv.dtype}"
        )
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_Linv_np = torch_Linv.detach().cpu().numpy()
        
        # Compare the matrices with looser tolerance due to potential SVD fallback
        self.assertTrue(
            np.allclose(np_Linv, torch_Linv_np, rtol=self.rtol_looser, atol=self.atol_looser),
            "Linv matrices are not equivalent within tolerance"
        )
    
    def test_sf_basic(self):
        """Test basic structure_factors_batch equivalence."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Prepare identical input arguments
        np_q_grid = np_model.q_grid[:10]  # Use a subset for faster testing
        torch_q_grid = torch.tensor(np_q_grid, dtype=torch.float64, device=self.device)
        
        np_xyz = np_model.model.xyz[0]
        torch_xyz = torch.tensor(np_xyz, dtype=torch.float64, device=self.device)
        
        np_ff_a = np_model.model.ff_a[0]
        torch_ff_a = torch.tensor(np_ff_a, dtype=torch.float64, device=self.device)
        
        np_ff_b = np_model.model.ff_b[0]
        torch_ff_b = torch.tensor(np_ff_b, dtype=torch.float64, device=self.device)
        
        np_ff_c = np_model.model.ff_c[0]
        torch_ff_c = torch.tensor(np_ff_c, dtype=torch.float64, device=self.device)
        
        # Call the standalone functions
        np_sf = np_structure_factors_batch(
            np_q_grid, np_xyz, np_ff_a, np_ff_b, np_ff_c,
            U=None, compute_qF=False, project_on_components=None, sum_over_atoms=True
        )
        
        torch_sf = torch_structure_factors_batch(
            torch_q_grid, torch_xyz, torch_ff_a, torch_ff_b, torch_ff_c,
            U=None, compute_qF=False, project_on_components=None, sum_over_atoms=True
        )
        
        # Verify torch_sf properties
        self.assertIsInstance(torch_sf, torch.Tensor)
        self.assertEqual(torch_sf.dtype, torch.complex128)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_sf_np = torch_sf.detach().cpu().numpy()
        
        # Compare the results
        self.assertTrue(
            np.allclose(np_sf, torch_sf_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Basic structure factors are not equivalent within tolerance"
        )
    
    def test_sf_compute_qF(self):
        """Test structure_factors_batch with compute_qF=True."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Prepare identical input arguments
        np_q_grid = np_model.q_grid[:10]  # Use a subset for faster testing
        torch_q_grid = torch.tensor(np_q_grid, dtype=torch.float64, device=self.device)
        
        np_xyz = np_model.model.xyz[0]
        torch_xyz = torch.tensor(np_xyz, dtype=torch.float64, device=self.device)
        
        np_ff_a = np_model.model.ff_a[0]
        torch_ff_a = torch.tensor(np_ff_a, dtype=torch.float64, device=self.device)
        
        np_ff_b = np_model.model.ff_b[0]
        torch_ff_b = torch.tensor(np_ff_b, dtype=torch.float64, device=self.device)
        
        np_ff_c = np_model.model.ff_c[0]
        torch_ff_c = torch.tensor(np_ff_c, dtype=torch.float64, device=self.device)
        
        # Call the standalone functions with compute_qF=True
        np_sf = np_structure_factors_batch(
            np_q_grid, np_xyz, np_ff_a, np_ff_b, np_ff_c,
            U=None, compute_qF=True, project_on_components=None, sum_over_atoms=True
        )
        
        torch_sf = torch_structure_factors_batch(
            torch_q_grid, torch_xyz, torch_ff_a, torch_ff_b, torch_ff_c,
            U=None, compute_qF=True, project_on_components=None, sum_over_atoms=True
        )
        
        # Verify torch_sf properties
        self.assertIsInstance(torch_sf, torch.Tensor)
        self.assertEqual(torch_sf.dtype, torch.complex128)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_sf_np = torch_sf.detach().cpu().numpy()
        
        # Compare the results
        self.assertTrue(
            np.allclose(np_sf, torch_sf_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Structure factors with compute_qF=True are not equivalent within tolerance"
        )
    
    def test_sf_no_sum(self):
        """Test structure_factors_batch with sum_over_atoms=False."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Prepare identical input arguments
        np_q_grid = np_model.q_grid[:10]  # Use a subset for faster testing
        torch_q_grid = torch.tensor(np_q_grid, dtype=torch.float64, device=self.device)
        
        np_xyz = np_model.model.xyz[0]
        torch_xyz = torch.tensor(np_xyz, dtype=torch.float64, device=self.device)
        
        np_ff_a = np_model.model.ff_a[0]
        torch_ff_a = torch.tensor(np_ff_a, dtype=torch.float64, device=self.device)
        
        np_ff_b = np_model.model.ff_b[0]
        torch_ff_b = torch.tensor(np_ff_b, dtype=torch.float64, device=self.device)
        
        np_ff_c = np_model.model.ff_c[0]
        torch_ff_c = torch.tensor(np_ff_c, dtype=torch.float64, device=self.device)
        
        # Call the standalone functions with sum_over_atoms=False
        np_sf = np_structure_factors_batch(
            np_q_grid, np_xyz, np_ff_a, np_ff_b, np_ff_c,
            U=None, compute_qF=False, project_on_components=None, sum_over_atoms=False
        )
        
        torch_sf = torch_structure_factors_batch(
            torch_q_grid, torch_xyz, torch_ff_a, torch_ff_b, torch_ff_c,
            U=None, compute_qF=False, project_on_components=None, sum_over_atoms=False
        )
        
        # Verify torch_sf properties
        self.assertIsInstance(torch_sf, torch.Tensor)
        self.assertEqual(torch_sf.dtype, torch.complex128)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_sf_np = torch_sf.detach().cpu().numpy()
        
        # Compare the results
        self.assertTrue(
            np.allclose(np_sf, torch_sf_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Structure factors with sum_over_atoms=False are not equivalent within tolerance"
        )
    
    def test_sf_projected(self):
        """Test structure_factors_batch with projection matrix."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Prepare identical input arguments
        np_q_grid = np_model.q_grid[:10]  # Use a subset for faster testing
        torch_q_grid = torch.tensor(np_q_grid, dtype=torch.float64, device=self.device)
        
        np_xyz = np_model.model.xyz[0]
        torch_xyz = torch.tensor(np_xyz, dtype=torch.float64, device=self.device)
        
        np_ff_a = np_model.model.ff_a[0]
        torch_ff_a = torch.tensor(np_ff_a, dtype=torch.float64, device=self.device)
        
        np_ff_b = np_model.model.ff_b[0]
        torch_ff_b = torch.tensor(np_ff_b, dtype=torch.float64, device=self.device)
        
        np_ff_c = np_model.model.ff_c[0]
        torch_ff_c = torch.tensor(np_ff_c, dtype=torch.float64, device=self.device)
        
        # Get projection matrices
        np_project = np_model.Amat[0]
        torch_project = torch_model.Amat[0]
        
        # Call the standalone functions with projection
        np_sf = np_structure_factors_batch(
            np_q_grid, np_xyz, np_ff_a, np_ff_b, np_ff_c,
            U=None, compute_qF=True, project_on_components=np_project, sum_over_atoms=False
        )
        
        torch_sf = torch_structure_factors_batch(
            torch_q_grid, torch_xyz, torch_ff_a, torch_ff_b, torch_ff_c,
            U=None, compute_qF=True, project_on_components=torch_project, sum_over_atoms=False
        )
        
        # Verify torch_sf properties
        self.assertIsInstance(torch_sf, torch.Tensor)
        self.assertEqual(torch_sf.dtype, torch.complex128)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_sf_np = torch_sf.detach().cpu().numpy()
        
        # Compare the results
        self.assertTrue(
            np.allclose(np_sf, torch_sf_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Structure factors with projection are not equivalent within tolerance"
        )
    
    def test_sf_with_adp(self):
        """Test structure_factors_batch with atomic displacement parameters (ADPs)."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Prepare identical input arguments
        np_q_grid = np_model.q_grid[:10]  # Use a subset for faster testing
        torch_q_grid = torch.tensor(np_q_grid, dtype=torch.float64, device=self.device)
        
        np_xyz = np_model.model.xyz[0]
        torch_xyz = torch.tensor(np_xyz, dtype=torch.float64, device=self.device)
        
        np_ff_a = np_model.model.ff_a[0]
        torch_ff_a = torch.tensor(np_ff_a, dtype=torch.float64, device=self.device)
        
        np_ff_b = np_model.model.ff_b[0]
        torch_ff_b = torch.tensor(np_ff_b, dtype=torch.float64, device=self.device)
        
        np_ff_c = np_model.model.ff_c[0]
        torch_ff_c = torch.tensor(np_ff_c, dtype=torch.float64, device=self.device)
        
        # Create ADPs (uniform values for simplicity)
        np_adp = np.ones(len(np_xyz), dtype=np.float64) * 0.5
        torch_adp = torch.ones(len(torch_xyz), dtype=torch.float64, device=self.device) * 0.5
        
        # Call the standalone functions with ADPs
        np_sf = np_structure_factors_batch(
            np_q_grid, np_xyz, np_ff_a, np_ff_b, np_ff_c,
            U=np_adp, compute_qF=False, project_on_components=None, sum_over_atoms=True
        )
        
        torch_sf = torch_structure_factors_batch(
            torch_q_grid, torch_xyz, torch_ff_a, torch_ff_b, torch_ff_c,
            U=torch_adp, compute_qF=False, project_on_components=None, sum_over_atoms=True
        )
        
        # Verify torch_sf properties
        self.assertIsInstance(torch_sf, torch.Tensor)
        self.assertEqual(torch_sf.dtype, torch.complex128)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_sf_np = torch_sf.detach().cpu().numpy()
        
        # Compare the results
        self.assertTrue(
            np.allclose(np_sf, torch_sf_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Structure factors with ADPs are not equivalent within tolerance"
        )
    
    def test_sf_combined(self):
        """Test structure_factors_batch with all options combined."""
        # Create models
        np_model, torch_model = self._create_models()
        
        # Prepare identical input arguments
        np_q_grid = np_model.q_grid[:10]  # Use a subset for faster testing
        torch_q_grid = torch.tensor(np_q_grid, dtype=torch.float64, device=self.device)
        
        np_xyz = np_model.model.xyz[0]
        torch_xyz = torch.tensor(np_xyz, dtype=torch.float64, device=self.device)
        
        np_ff_a = np_model.model.ff_a[0]
        torch_ff_a = torch.tensor(np_ff_a, dtype=torch.float64, device=self.device)
        
        np_ff_b = np_model.model.ff_b[0]
        torch_ff_b = torch.tensor(np_ff_b, dtype=torch.float64, device=self.device)
        
        np_ff_c = np_model.model.ff_c[0]
        torch_ff_c = torch.tensor(np_ff_c, dtype=torch.float64, device=self.device)
        
        # Create ADPs
        np_adp = np.ones(len(np_xyz), dtype=np.float64) * 0.5
        torch_adp = torch.ones(len(torch_xyz), dtype=torch.float64, device=self.device) * 0.5
        
        # Get projection matrices
        np_project = np_model.Amat[0]
        torch_project = torch_model.Amat[0]
        
        # Call the standalone functions with all options
        np_sf = np_structure_factors_batch(
            np_q_grid, np_xyz, np_ff_a, np_ff_b, np_ff_c,
            U=np_adp, compute_qF=True, project_on_components=np_project, sum_over_atoms=False
        )
        
        torch_sf = torch_structure_factors_batch(
            torch_q_grid, torch_xyz, torch_ff_a, torch_ff_b, torch_ff_c,
            U=torch_adp, compute_qF=True, project_on_components=torch_project, sum_over_atoms=False
        )
        
        # Verify torch_sf properties
        self.assertIsInstance(torch_sf, torch.Tensor)
        self.assertEqual(torch_sf.dtype, torch.complex128)
        
        # Convert PyTorch tensor to NumPy array for comparison
        torch_sf_np = torch_sf.detach().cpu().numpy()
        
        # Compare the results
        self.assertTrue(
            np.allclose(np_sf, torch_sf_np, rtol=self.rtol_tight, atol=self.atol_tight),
            "Structure factors with all options are not equivalent within tolerance"
        )


if __name__ == "__main__":
    unittest.main()
