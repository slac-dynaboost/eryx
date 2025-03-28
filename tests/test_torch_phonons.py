import unittest
import numpy as np
import torch
from tests.torch_test_base import TorchComponentTestCase
from tests.test_helpers.component_tests import PhononTests

class TestTorchPhonons(TorchComponentTestCase):
    """Test suite for PyTorch phonon calculation components."""
    
    def setUp(self):
        """Set up test environment with default configuration."""
        super().setUp()
        # Use smaller test parameters for faster testing
        self.test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],  # Smaller grid for faster testing
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    
    def test_compute_gnm_phonons(self):
        """Test the compute_gnm_phonons method implementation."""
        # Create models with small test parameters for speed
        self.create_models()
        
        # Execute compute_gnm_phonons on both models
        self.np_model.compute_gnm_phonons()
        self.torch_model.compute_gnm_phonons()
        
        # Check V (eigenvectors) has correct shape
        np_v_shape = self.np_model.V.shape
        torch_v_shape = tuple(self.torch_model.V.shape)
        
        self.assertEqual(
            np_v_shape, torch_v_shape,
            f"Eigenvector (V) shapes don't match: NP={np_v_shape}, Torch={torch_v_shape}"
        )
        
        # Check Winv (eigenvalues) has correct shape
        np_winv_shape = self.np_model.Winv.shape
        torch_winv_shape = tuple(self.torch_model.Winv.shape)
        
        self.assertEqual(
            np_winv_shape, torch_winv_shape,
            f"Eigenvalue (Winv) shapes don't match: NP={np_winv_shape}, Torch={torch_winv_shape}"
        )
        
        # Test eigenvalues for each k-vector
        # For smaller test case, get dimensions
        h_dim = self.test_params['hsampling'][2]
        k_dim = self.test_params['ksampling'][2]
        l_dim = self.test_params['lsampling'][2]
        
        # Test several k-vectors (first, middle, last)
        for dh in range(0, h_dim, max(1, h_dim-1)):
            for dk in range(0, k_dim, max(1, k_dim-1)):
                for dl in range(0, l_dim, max(1, l_dim-1)):
                    with self.subTest(f"k-vector_({dh},{dk},{dl})"):
                        # Compare eigenvalues (Winv)
                        np_winv = self.np_model.Winv[dh, dk, dl]
                        torch_winv = self.torch_model.Winv[dh, dk, dl]
                        
                        # Use specialized comparison for eigenvalues
                        results = PhononTests.compare_eigenvalues(
                            np_winv, torch_winv, rtol=1e-4, atol=1e-6
                        )
                        
                        # Check success
                        self.assertTrue(
                            results['success'],
                            f"Eigenvalue comparison failed for k-vector ({dh},{dk},{dl}): {results}"
                        )
                        
                        # Compare eigenvectors (V)
                        np_v = self.np_model.V[dh, dk, dl]
                        torch_v = self.torch_model.V[dh, dk, dl]
                        
                        # Use specialized comparison for eigenvectors
                        results = PhononTests.compare_eigenvectors(
                            np_v, torch_v, rtol=1e-4, atol=1e-6
                        )
                        
                        # Check success
                        self.assertTrue(
                            results['success'],
                            f"Eigenvector comparison failed for k-vector ({dh},{dk},{dl}): {results}"
                        )
    
    def test_compute_covariance_matrix(self):
        """Test the compute_covariance_matrix method implementation."""
        # Create models
        self.create_models()
        
        # Execute compute_gnm_phonons first (required by compute_covariance_matrix)
        self.np_model.compute_gnm_phonons()
        self.torch_model.compute_gnm_phonons()
        
        # Execute compute_covariance_matrix
        self.np_model.compute_covariance_matrix()
        self.torch_model.compute_covariance_matrix()
        
        # Check covariance matrices have same shape
        np_covar_shape = self.np_model.covar.shape
        torch_covar_shape = tuple(self.torch_model.covar.shape)
        
        self.assertEqual(
            np_covar_shape, torch_covar_shape,
            f"Covariance matrix shapes don't match: NP={np_covar_shape}, Torch={torch_covar_shape}"
        )
        
        # Check covariance values match
        # Convert PyTorch tensor to NumPy
        torch_covar = self.torch_model.covar.detach().cpu().numpy()
        
        # Compare with higher tolerance for covariance (accumulated differences)
        self.assert_tensors_equal(
            self.np_model.covar, torch_covar,
            rtol=1e-3, atol=1e-6,
            msg="Covariance matrix values don't match"
        )
        
        # Check ADP values match
        np_adp = self.np_model.ADP
        torch_adp = self.torch_model.ADP.detach().cpu().numpy()
        
        self.assert_tensors_equal(
            np_adp, torch_adp,
            rtol=1e-3, atol=1e-6,
            msg="ADP values don't match"
        )

if __name__ == '__main__':
    unittest.main()
