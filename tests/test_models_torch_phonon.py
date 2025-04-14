import unittest
import os
import glob
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon
from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
from eryx.autotest.test_helpers import load_test_state, build_test_object, ensure_tensor, verify_gradient_flow
from tests.test_helpers.component_tests import PhononTests
from tests.torch_test_utils import TensorComparison

class TestOnePhononPhonon(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
        # Create minimal test model
        self.model = self._create_test_model()
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes for phonon methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes
        model.device = self.device
        model.n_asu = 2
        model.n_cell = 3
        model.n_atoms_per_asu = 4
        model.n_dof_per_asu = 6
        model.n_dof_per_asu_actual = 12  # n_atoms_per_asu * 3
        model.n_dof_per_cell = 12  # n_asu * n_dof_per_asu
        model.id_cell_ref = 0
        
        # Create other necessary attributes
        model.hsampling = (0, 5, 2)
        model.ksampling = (0, 5, 2)
        model.lsampling = (0, 5, 2)
        
        # Create sample matrices
        model.Amat = torch.zeros((model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu), 
                                device=self.device)
        model.Linv = torch.eye(model.n_dof_per_cell, device=self.device)
        
        # Create k-vectors for testing
        model.kvec = torch.zeros((model.hsampling[2], model.ksampling[2], model.lsampling[2], 3), 
                                device=self.device)
        model.kvec_norm = torch.zeros((model.hsampling[2], model.ksampling[2], model.lsampling[2], 1), 
                                     device=self.device)
        
        # Sample methods that would be called
        model.id_to_hkl = lambda cell_id: [0, 0, 0]
        model.get_unitcell_origin = lambda unit_cell: torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        return model
    
    def create_models(self, test_params=None):
        """Create NumPy and PyTorch models for comparative testing."""
        # Import NumPy model for comparison
        from eryx.models import OnePhonon as NumpyOnePhonon
        
        # Default test parameters
        self.test_params = test_params or {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Create NumPy model for reference
        self.np_model = NumpyOnePhonon(**self.test_params)
        
        # Create PyTorch model
        self.torch_model = OnePhonon(
            **self.test_params,
            device=self.device
        )
    
    def test_compute_hessian(self):
        """Test compute_hessian method using state-based approach."""
        try:
            # Load before state
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_hessian"
            )
            
            # Load after state for expected output
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_hessian",
                before=False
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*compute_hessian*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Call the method under test
        hessian = model.compute_hessian()
        
        # Verify result properties
        self.assertIsInstance(hessian, torch.Tensor, "Result should be a tensor")
        self.assertEqual(hessian.dtype, torch.complex64, "Result should be complex64")
        
        # Expected shape based on model attributes
        expected_shape = (model.n_asu, model.n_dof_per_asu, 
                          model.n_cell, model.n_asu, model.n_dof_per_asu)
        self.assertEqual(hessian.shape, expected_shape, "Hessian has incorrect shape")
        
        # Get expected output from after state
        expected_hessian = after_state.get('hessian')
        if expected_hessian is not None:
            # Convert to tensor for comparison
            expected_hessian = ensure_tensor(expected_hessian, device='cpu')
            
            # Compare with expected output
            hessian_np = hessian.detach().cpu().numpy()
            expected_np = expected_hessian.detach().cpu().numpy() if isinstance(expected_hessian, torch.Tensor) else expected_hessian
            
            self.assertTrue(np.allclose(hessian_np, expected_np, rtol=1e-5, atol=1e-8),
                          "compute_hessian doesn't match ground truth")

    def test_compute_gnm_phonons_state_based(self):
        """Test compute_gnm_phonons method using state-based approach."""
        try:
            # Load before state
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_gnm_phonons"
            )
            
            # Load after state for expected output
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_gnm_phonons",
                before=False
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*compute_gnm_phonons*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Call the method under test
        model.compute_gnm_phonons()
        
        # Verify result properties
        self.assertIsInstance(model.V, torch.Tensor, "V should be a tensor")
        self.assertEqual(model.V.dtype, torch.complex64, "V should be complex64")
        self.assertIsInstance(model.Winv, torch.Tensor, "Winv should be a tensor")
        self.assertEqual(model.Winv.dtype, torch.complex64, "Winv should be complex64")
        
        # Get expected output from after state
        V_expected = after_state.get('V')
        Winv_expected = after_state.get('Winv')
        
        if V_expected is not None and Winv_expected is not None:
            # Convert to tensors for comparison
            V_expected_np = ensure_tensor(V_expected, device='cpu').detach().cpu().numpy()
            Winv_expected_np = ensure_tensor(Winv_expected, device='cpu').detach().cpu().numpy()
            
            # Get actual results
            V_actual_np = model.V.detach().cpu().numpy()
            Winv_actual_np = model.Winv.detach().cpu().numpy()
            
            # Compare Winv with handling for NaN values
            # First check NaN patterns match
            np_nans = np.isnan(Winv_expected_np)
            torch_nans = np.isnan(Winv_actual_np)
            self.assertTrue(np.array_equal(np_nans, torch_nans), 
                          "NaN patterns in Winv don't match")
            
            # Compare non-NaN values
            mask = ~np_nans
            if np.any(mask):
                self.assertTrue(np.allclose(Winv_expected_np[mask], Winv_actual_np[mask], 
                                          rtol=1e-4, atol=1e-6),
                              "Non-NaN values in Winv don't match")
            
            # Compare V with handling for eigenvector ambiguity
            # Compare absolute values to handle sign/phase ambiguity
            V_expected_abs = np.abs(V_expected_np)
            V_actual_abs = np.abs(V_actual_np)
            
            self.assertTrue(np.allclose(V_expected_abs, V_actual_abs, 
                                      rtol=1e-4, atol=1e-6),
                          "Absolute values of eigenvectors V don't match")
            
            # Additional verification: check orthogonality
            for i in range(V_actual_np.shape[0]):
                V_i = V_actual_np[i]
                product = np.matmul(np.conjugate(V_i.T), V_i)
                identity = np.eye(V_i.shape[1])
                self.assertTrue(np.allclose(product, identity, rtol=1e-3, atol=1e-3),
                              f"Eigenvectors at index {i} are not orthogonal")

    def test_log_completeness(self):
        """Verify phonon-related logs exist and contain required attributes."""
        if not hasattr(self, 'verify_logs') or not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify phonon method logs
        self.verify_required_logs(self.module_name, "compute_gnm_phonons", ["V", "Winv"])
        self.verify_required_logs(self.module_name, "compute_hessian", ["hessian"])
    
if __name__ == '__main__':
    unittest.main()
