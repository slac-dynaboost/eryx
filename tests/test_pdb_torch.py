import unittest
import os
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.pdb_torch import GaussianNetworkModel, sym_str_as_matrix
from eryx.autotest.test_helpers import load_test_state, build_test_object, ensure_tensor, verify_gradient_flow

class TestGaussianNetworkModel(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        
        # Set module name for log paths
        self.module_name = "eryx.pdb"
        self.class_name = "GaussianNetworkModel"
    
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
        model = build_test_object(GaussianNetworkModel, before_state, device=self.device)
        
        # Call the method under test
        hessian = model.compute_hessian()
        
        # Verify result properties
        self.assertIsInstance(hessian, torch.Tensor, "Result should be a tensor")
        self.assertEqual(hessian.dtype, torch.complex64, "Result should be complex64")
        
        # Expected shape based on model attributes
        expected_shape = (model.n_asu, model.n_atoms_per_asu, 
                          model.n_cell, model.n_asu, model.n_atoms_per_asu)
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
    
    def test_compute_K(self):
        """Test compute_K method using state-based approach."""
        try:
            # Load before state
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_K"
            )
            
            # Load after state for expected output
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_K",
                before=False
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*compute_K*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(GaussianNetworkModel, before_state, device=self.device)
        
        # Get hessian and kvec from before state
        hessian = before_state.get('hessian')
        kvec = before_state.get('kvec')
        
        if hessian is None or kvec is None:
            self.skipTest("Missing required inputs (hessian or kvec) in before state")
            return
        
        # Convert to tensors
        hessian = ensure_tensor(hessian, device=self.device)
        kvec = ensure_tensor(kvec, device=self.device)
        
        # Call the method under test
        Kmat = model.compute_K(hessian, kvec)
        
        # Verify result properties
        self.assertIsInstance(Kmat, torch.Tensor, "Result should be a tensor")
        self.assertEqual(Kmat.dtype, torch.complex64, "Result should be complex64")
        
        # Expected shape based on model attributes
        expected_shape = (model.n_asu, model.n_atoms_per_asu, model.n_asu, model.n_atoms_per_asu)
        self.assertEqual(Kmat.shape, expected_shape, "K matrix has incorrect shape")
        
        # Get expected output from after state
        expected_K = after_state.get('K')
        if expected_K is not None:
            # Convert to tensor for comparison
            expected_K = ensure_tensor(expected_K, device='cpu')
            
            # Compare with expected output
            K_np = Kmat.detach().cpu().numpy()
            expected_np = expected_K.detach().cpu().numpy() if isinstance(expected_K, torch.Tensor) else expected_K
            
            self.assertTrue(np.allclose(K_np, expected_np, rtol=1e-5, atol=1e-8),
                          "compute_K doesn't match ground truth")
    
    def test_compute_Kinv(self):
        """Test compute_Kinv method using state-based approach."""
        try:
            # Load before state
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_Kinv"
            )
            
            # Load after state for expected output
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_Kinv",
                before=False
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*compute_Kinv*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(GaussianNetworkModel, before_state, device=self.device)
        
        # Get hessian and kvec from before state
        hessian = before_state.get('hessian')
        kvec = before_state.get('kvec')
        reshape = before_state.get('reshape', True)
        
        if hessian is None:
            self.skipTest("Missing required input (hessian) in before state")
            return
        
        # Convert to tensors
        hessian = ensure_tensor(hessian, device=self.device)
        kvec = ensure_tensor(kvec, device=self.device) if kvec is not None else None
        
        # Call the method under test
        Kinv = model.compute_Kinv(hessian, kvec, reshape)
        
        # Verify result properties
        self.assertIsInstance(Kinv, torch.Tensor, "Result should be a tensor")
        self.assertEqual(Kinv.dtype, torch.complex64, "Result should be complex64")
        
        # Get expected output from after state
        expected_Kinv = after_state.get('Kinv')
        if expected_Kinv is not None:
            # Convert to tensor for comparison
            expected_Kinv = ensure_tensor(expected_Kinv, device='cpu')
            
            # Compare with expected output
            Kinv_np = Kinv.detach().cpu().numpy()
            expected_np = expected_Kinv.detach().cpu().numpy() if isinstance(expected_Kinv, torch.Tensor) else expected_Kinv
            
            self.assertTrue(np.allclose(Kinv_np, expected_np, rtol=1e-5, atol=1e-8),
                          "compute_Kinv doesn't match ground truth")

if __name__ == '__main__':
    unittest.main()
