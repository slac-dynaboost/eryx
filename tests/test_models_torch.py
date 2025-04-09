import os
import unittest
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon


class TestMatrixConstruction(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
    def test_build_A_state_based(self):
        """Test _build_A using state-based approach."""
        # Import test helpers
        from eryx.autotest.test_helpers import (
            load_test_state,
            build_test_object,
            ensure_tensor
        )
        
        try:
            # Load before state using the helper function
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "_build_A"
            )
        except FileNotFoundError as e:
            # Handle missing logs gracefully
            import glob
            available_logs = glob.glob("logs/*build_A*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Call the method under test
        model._build_A()
        
        # Verify results - check Amat tensor
        self.assertTrue(hasattr(model, 'Amat'), "Amat not created")
        expected_shape = (model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu)
        self.assertEqual(model.Amat.shape, expected_shape)
        self.assertTrue(model.Amat.requires_grad, "Amat should require gradients")
        
        # Load after state for comparison
        try:
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "_build_A",
                before=False
            )
        except FileNotFoundError as e:
            self.skipTest(f"Could not find after state log: {e}")
            return
        
        # Get expected tensor from after state
        expected_amat = after_state.get('Amat')
        
        # Check if expected_amat exists
        if expected_amat is None:
            self.skipTest("Expected Amat not found in after state log")
            return
            
        expected_amat = ensure_tensor(expected_amat, device='cpu')
        
        # Convert model tensor to numpy for comparison
        amat_numpy = model.Amat.detach().cpu().numpy()
        expected_amat_numpy = expected_amat.detach().cpu().numpy() if isinstance(expected_amat, torch.Tensor) else expected_amat
        
        # Compare with appropriate tolerances for float64 calculations
        tolerances = {'rtol': 1e-4, 'atol': 1e-5}
        
        # Print differences for debugging
        max_diff = np.max(np.abs(amat_numpy - expected_amat_numpy))
        print(f"Maximum difference: {max_diff}")
        
        # Verify tensors match expected values
        self.assertTrue(
            np.allclose(
                amat_numpy, 
                expected_amat_numpy, 
                rtol=tolerances['rtol'], 
                atol=tolerances['atol']
            ),
            "Amat values don't match expected"
        )
        
    def test_build_M_allatoms_state_based(self):
        """Test _build_M_allatoms using state-based approach."""
        # Import test helpers
        from eryx.autotest.test_helpers import (
            load_test_state,
            build_test_object,
            ensure_tensor
        )
        
        try:
            # Load before state using the helper function
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "_build_M_allatoms"
            )
        except FileNotFoundError as e:
            # Handle missing logs gracefully
            import glob
            available_logs = glob.glob("logs/*build_M_allatoms*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Call the method under test
        result = model._build_M_allatoms()
        
        # Verify basic properties of result
        expected_shape = (model.n_asu, model.n_dof_per_asu_actual,
                         model.n_asu, model.n_dof_per_asu_actual)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(result.requires_grad, "Result should require gradients")
        self.assertTrue(torch.all(result >= 0), "Mass matrix should be non-negative")
        
        # Load after state for comparison
        try:
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "_build_M_allatoms",
                before=False
            )
        except FileNotFoundError as e:
            self.skipTest(f"Could not find after state log: {e}")
            return
        
        # Get expected result from after state
        expected_result = after_state.get('return')
        
        # Check if expected_result exists
        if expected_result is None:
            # Skip comparison if no expected result is found
            print("No expected result found in after state log, skipping comparison")
            return
            
        expected_result = ensure_tensor(expected_result, device='cpu')
        
        # Convert model tensor to numpy for comparison
        result_numpy = result.detach().cpu().numpy()
        expected_result_numpy = expected_result.detach().cpu().numpy() if isinstance(expected_result, torch.Tensor) else expected_result
        
        # Compare with appropriate tolerances for float64 calculations
        tolerances = {'rtol': 1e-4, 'atol': 1e-5}
        
        # Print differences for debugging
        max_diff = np.max(np.abs(result_numpy - expected_result_numpy))
        print(f"Maximum difference: {max_diff}")
        
        # Verify tensors match expected values
        self.assertTrue(
            np.allclose(
                result_numpy, 
                expected_result_numpy, 
                rtol=tolerances['rtol'], 
                atol=tolerances['atol']
            ),
            "M_allatoms values don't match expected"
        )
        
    def test_project_M_state_based(self):
        """Test _project_M using state-based approach."""
        # Import test helpers
        from eryx.autotest.test_helpers import (
            load_test_state,
            build_test_object,
            ensure_tensor
        )
        
        try:
            # Load before state using the helper function
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "_project_M"
            )
        except FileNotFoundError as e:
            # Handle missing logs gracefully
            import glob
            available_logs = glob.glob("logs/*project_M*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Ensure M_allatoms is properly initialized
        if not hasattr(model, 'M_allatoms') or model.M_allatoms is None:
            # Try to find M_allatoms in the state
            if 'M_allatoms' in before_state:
                model.M_allatoms = ensure_tensor(before_state['M_allatoms'], device=model.device)
            else:
                # Create a dummy tensor with the right shape as a fallback
                model.M_allatoms = torch.ones(
                    (model.n_asu, model.n_dof_per_asu_actual, model.n_asu, model.n_dof_per_asu_actual),
                    device=model.device, 
                    dtype=torch.float64, 
                    requires_grad=True
                )
        
        # Call the method under test
        result = model._project_M(model.M_allatoms)
        
        # Verify basic properties of result
        expected_shape = (model.n_asu, model.n_dof_per_asu,
                         model.n_asu, model.n_dof_per_asu)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(result.requires_grad, "Result should require gradients")
        
        # Load after state for comparison
        try:
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "_project_M",
                before=False
            )
        except FileNotFoundError as e:
            self.skipTest(f"Could not find after state log: {e}")
            return
        
        # Get expected result from after state
        expected_result = after_state.get('return')
        
        # Check if expected_result exists
        if expected_result is None:
            # Skip comparison if no expected result is found
            print("No expected result found in after state log, skipping comparison")
            return
            
        expected_result = ensure_tensor(expected_result, device='cpu')
        
        # Convert model tensor to numpy for comparison
        result_numpy = result.detach().cpu().numpy()
        expected_result_numpy = expected_result.detach().cpu().numpy() if isinstance(expected_result, torch.Tensor) else expected_result
        
        # Compare with appropriate tolerances for float64 calculations
        tolerances = {'rtol': 1e-4, 'atol': 1e-5}
        
        # Print differences for debugging
        max_diff = np.max(np.abs(result_numpy - expected_result_numpy))
        print(f"Maximum difference: {max_diff}")
        
        # Verify tensors match expected values
        self.assertTrue(
            np.allclose(
                result_numpy, 
                expected_result_numpy, 
                rtol=tolerances['rtol'], 
                atol=tolerances['atol']
            ),
            "Projected M values don't match expected"
        )

    def test_log_completeness(self):
        """Verify matrix construction logs exist and contain required attributes."""
        if not hasattr(self, 'verify_logs') or not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify matrix construction logs
        self.verify_required_logs(self.module_name, "_build_A", ["Amat"])
        self.verify_required_logs(self.module_name, "_build_M", ["Linv"])
        self.verify_required_logs(self.module_name, "_build_M_allatoms", [])
        self.verify_required_logs(self.module_name, "_project_M", [])

if __name__ == '__main__':
    unittest.main()
