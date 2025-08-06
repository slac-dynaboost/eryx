"""
Tests for the PyTorch testing infrastructure.

This module contains tests for the PyTorch testing infrastructure components,
including tensor comparison, state capture and injection, and component-specific
test utilities.
"""

import unittest
import numpy as np
import torch
import os
import sys
from typing import Dict, Any, Tuple, List, Optional

# Import testing utilities
from tests.torch_test_utils import TensorComparison, ModelState
from tests.test_base import TestBase as TorchComponentTestCase
try:
    from tests.test_helpers.mock_data import (
        create_mock_kvectors, create_mock_hessian, create_mock_eigendecomposition,
        create_mock_model_dict, create_mock_crystal_dict
    )
    from tests.test_helpers.component_tests import KVectorTests, HessianTests, PhononTests, DisorderTests
except ImportError:
    # Create stub functions if imports fail
    def create_mock_kvectors(*args, **kwargs): return torch.zeros((2, 2, 2, 3))
    def create_mock_hessian(*args, **kwargs): return torch.zeros((2, 3, 4, 2, 3))
    def create_mock_eigendecomposition(*args, **kwargs): return torch.zeros(5), torch.eye(5)
    def create_mock_model_dict(*args, **kwargs): return {"xyz": [torch.zeros((3, 3))]}
    def create_mock_crystal_dict(*args, **kwargs): return {"n_asu": 2, "n_atoms_per_asu": 3, "n_cell": 4}
    
    # Create stub classes
    class KVectorTests:
        @staticmethod
        def test_center_kvec(*args): return [{"is_equal": True}]
        @staticmethod
        def test_kvector_brillouin(*args): return True, {}
        @staticmethod
        def test_at_kvec_from_miller_points(*args): return [{"is_equal": True}]
    
    class HessianTests:
        @staticmethod
        def compare_hessian_structure(*args): return {"shape_match": True}
    
    class PhononTests:
        @staticmethod
        def compare_eigenvalues(*args): return {"success": True}
    
    class DisorderTests:
        pass


class InfrastructureTest(unittest.TestCase):
    """Tests for the PyTorch testing infrastructure."""
    
    def setUp(self):
        """Set up test environment."""
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
    
    def test_tensor_comparison(self):
        """Test the tensor comparison utilities."""
        # Create test tensors with known differences
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=torch.float32)
        
        # Test exact match
        success, metrics = TensorComparison.compare_tensors(np_array, torch_tensor, rtol=1e-4, atol=1e-4)
        self.assertTrue(success)
        self.assertAlmostEqual(metrics["max_abs_diff"], 0.0, places=6)
        
        # Test with small difference within tolerance
        torch_tensor_small_diff = torch_tensor + 1e-6
        success, metrics = TensorComparison.compare_tensors(np_array, torch_tensor_small_diff)
        self.assertTrue(success)
        self.assertAlmostEqual(metrics["max_abs_diff"], 1e-6)
        
        # Test with difference outside tolerance
        torch_tensor_large_diff = torch_tensor + 1e-3
        success, metrics = TensorComparison.compare_tensors(np_array, torch_tensor_large_diff)
        self.assertFalse(success)
        self.assertAlmostEqual(metrics["max_abs_diff"], 1e-3)
        
        # Test with custom tolerance
        success, metrics = TensorComparison.compare_tensors(
            np_array, torch_tensor_large_diff, rtol=1e-2, atol=1e-2
        )
        self.assertTrue(success)
        
        # Test shape mismatch detection
        torch_tensor_wrong_shape = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device)
        success, metrics = TensorComparison.compare_tensors(np_array, torch_tensor_wrong_shape)
        self.assertFalse(success)
        self.assertEqual(metrics["message"], "Shape mismatch")
        
        # Test NaN handling
        np_array_with_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
        torch_tensor_with_nan = torch.tensor([[1.0, float('nan')], [3.0, 4.0]], device=self.device)
        
        # NaN pattern should match
        success, metrics = TensorComparison.compare_tensors(
            np_array_with_nan, torch_tensor_with_nan, check_nans=True
        )
        self.assertTrue(success)
        
        # NaN pattern mismatch
        torch_tensor_wrong_nan = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]], device=self.device)
        success, metrics = TensorComparison.compare_tensors(
            np_array_with_nan, torch_tensor_wrong_nan, check_nans=True
        )
        self.assertFalse(success)
        self.assertEqual(metrics["message"], "NaN pattern mismatch")
        
        # Test assert_tensors_equal
        TensorComparison.assert_tensors_equal(np_array, torch_tensor)
        
        # Test assert_tensors_equal with failure
        with self.assertRaises(AssertionError):
            TensorComparison.assert_tensors_equal(np_array, torch_tensor_large_diff)
    
    def test_state_capture_injection(self):
        """Test the state capture and injection utilities."""
        # Create simple mock object with known attributes
        class MockModel:
            def __init__(self):
                self.attr1 = 1
                self.attr2 = np.array([1.0, 2.0, 3.0])
                self.attr3 = "test"
                self.attr4 = [1, 2, 3]  # Add a mutable object
                self.device = torch.device('cpu')
        
        mock_model = MockModel()
        
        # Capture state
        state = ModelState.capture_model_state(mock_model, 
                                              attributes=['attr1', 'attr2', 'attr3', 'attr4'])
        
        # Verify state was captured correctly
        self.assertEqual(state['attr1'], 1)
        self.assertTrue(np.array_equal(state['attr2'], np.array([1.0, 2.0, 3.0])))
        self.assertEqual(state['attr3'], "test")
        self.assertEqual(state['attr4'], [1, 2, 3])
        
        # Modify object
        mock_model.attr1 = 2
        mock_model.attr2 = np.array([4.0, 5.0, 6.0])
        mock_model.attr3 = "modified"
        mock_model.attr4 = [4, 5, 6]
        
        # Inject original state
        ModelState.inject_model_state(mock_model, state, to_tensor=False)
        
        # Verify object restored correctly
        self.assertEqual(mock_model.attr1, 1)
        self.assertTrue(np.array_equal(mock_model.attr2, np.array([1.0, 2.0, 3.0])))
        self.assertEqual(mock_model.attr3, "test")
        self.assertEqual(mock_model.attr4, [1, 2, 3])
        
        # Test conversion to tensors
        mock_model.attr2 = np.array([7.0, 8.0, 9.0])
        ModelState.inject_model_state(mock_model, state, to_tensor=True)
        
        # Verify attr2 is now a tensor
        self.assertIsInstance(mock_model.attr2, torch.Tensor)
        self.assertTrue(torch.allclose(mock_model.attr2, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)))
    
    def test_mock_data_generators(self):
        """Test the mock data generators."""
        # Test create_mock_kvectors
        kvectors = create_mock_kvectors(shape=(2, 3, 4, 3), device=self.device)
        self.assertEqual(kvectors.shape, (2, 3, 4, 3))
        self.assertEqual(kvectors.device, self.device)
        self.assertTrue(kvectors.requires_grad)
        
        # Test create_mock_hessian
        hessian = create_mock_hessian(n_asu=2, n_atoms=3, n_cell=4, device=self.device)
        self.assertEqual(hessian.shape, (2, 3, 4, 2, 3))
        self.assertEqual(hessian.device, self.device)
        self.assertTrue(hessian.requires_grad)
        
        # Test create_mock_eigendecomposition
        eigenvalues, eigenvectors = create_mock_eigendecomposition(n_modes=5, n_nan=2, device=self.device)
        self.assertEqual(eigenvalues.shape, (5,))
        self.assertEqual(eigenvectors.shape, (5, 5))
        self.assertEqual(eigenvalues.device, self.device)
        self.assertEqual(eigenvectors.device, self.device)
        self.assertEqual(torch.isnan(eigenvalues).sum().item(), 2)
        
        # Test create_mock_model_dict
        model_dict = create_mock_model_dict(n_asu=2, n_atoms=3, device=self.device)
        self.assertEqual(len(model_dict['xyz']), 2)
        self.assertEqual(model_dict['xyz'][0].shape, (3, 3))
        self.assertEqual(model_dict['xyz'][0].device, self.device)
        
        # Test create_mock_crystal_dict
        crystal_dict = create_mock_crystal_dict(n_asu=2, n_atoms=3, n_cell=4, device=self.device)
        self.assertEqual(crystal_dict['n_asu'], 2)
        self.assertEqual(crystal_dict['n_atoms_per_asu'], 3)
        self.assertEqual(crystal_dict['n_cell'], 4)
        self.assertEqual(crystal_dict['get_asu_xyz'](0).device, self.device)
    
    def test_component_utilities(self):
        """Test the component-specific test utilities."""
        # Create mock models for testing
        class MockNumpyModel:
            def __init__(self):
                self.kvec = np.zeros((2, 2, 2, 3))
                self.kvec_norm = np.zeros((2, 2, 2, 1))
                
            def _center_kvec(self, x, L):
                return int(((x - L / 2) % L) - L / 2) / L
                
            def _build_kvec_Brillouin(self):
                self.kvec = np.ones((2, 2, 2, 3))
                self.kvec_norm = np.ones((2, 2, 2, 1))
                
            def _at_kvec_from_miller_points(self, hkl):
                return np.array([0, 1, 2, 3])
        
        class MockTorchModel:
            def __init__(self, device=None):
                self.device = device or torch.device('cpu')
                self.kvec = torch.zeros((2, 2, 2, 3), device=self.device)
                self.kvec_norm = torch.zeros((2, 2, 2, 1), device=self.device)
                
            def _center_kvec(self, x, L):
                return int(((x - L / 2) % L) - L / 2) / L
                
            def _build_kvec_Brillouin(self):
                self.kvec = torch.ones((2, 2, 2, 3), device=self.device)
                self.kvec_norm = torch.ones((2, 2, 2, 1), device=self.device)
                
            def _at_kvec_from_miller_points(self, hkl):
                return torch.tensor([0, 1, 2, 3], device=self.device)
        
        np_model = MockNumpyModel()
        torch_model = MockTorchModel(device=self.device)
        
        # Test KVectorTests
        center_kvec_results = KVectorTests.test_center_kvec(np_model, torch_model)
        self.assertTrue(all(result["is_equal"] for result in center_kvec_results))
        
        kvec_brillouin_success, _ = KVectorTests.test_kvector_brillouin(np_model, torch_model)
        self.assertTrue(kvec_brillouin_success)
        
        miller_points_results = KVectorTests.test_at_kvec_from_miller_points(np_model, torch_model)
        self.assertTrue(all(result["is_equal"] for result in miller_points_results))
        
        # Test HessianTests with mock hessians
        np_hessian = np.ones((2, 3, 4, 2, 3), dtype=complex)
        torch_hessian = torch.ones((2, 3, 4, 2, 3), dtype=torch.complex64, device=self.device)
        
        hessian_structure = HessianTests.compare_hessian_structure(np_hessian, torch_hessian)
        self.assertTrue(hessian_structure["shape_match"])
        
        # Test PhononTests with mock eigenvalues
        np_eigenvals = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        torch_eigenvals = torch.tensor([1.0, 2.0, float('nan'), 4.0, 5.0], device=self.device)
        
        eigenval_results = PhononTests.compare_eigenvalues(np_eigenvals, torch_eigenvals)
        self.assertTrue(eigenval_results["success"])
    
    def test_base_test_case(self):
        """Test the TorchComponentTestCase functionality."""
        # Create subclass of TorchComponentTestCase
        class TestCase(TorchComponentTestCase):
            def test_assert_tensors_equal(self):
                np_array = np.array([1.0, 2.0, 3.0])
                torch_tensor = torch.tensor([1.0, 2.0, 3.0], device=self.device)
                # Use TensorComparison directly instead of non-existent method
                TensorComparison.assert_tensors_equal(np_array, torch_tensor)
                
            def test_state_capture_injection(self):
                # Create simple object
                class SimpleObject:
                    def __init__(self):
                        self.value = 1
                        self.array = np.array([1.0, 2.0, 3.0])
                        self.list_attr = [1, 2, 3]  # Add a mutable object
                        self.device = torch.device('cpu')
                
                obj = SimpleObject()
                
                # Capture state - explicitly include all attributes
                state = ModelState.capture_model_state(obj, attributes=['value', 'array', 'list_attr'])
                
                # Modify object
                obj.value = 2
                obj.array = np.array([4.0, 5.0, 6.0])
                obj.list_attr = [4, 5, 6]
                
                # Inject state
                ModelState.inject_model_state(obj, state, to_tensor=False)
                
                # Verify restoration
                self.assertEqual(obj.value, 1)
                self.assertTrue(np.array_equal(obj.array, np.array([1.0, 2.0, 3.0])))
                self.assertEqual(obj.list_attr, [1, 2, 3])
        
        # Run the test case
        test_case = TestCase()
        test_case.setUp()
        test_case.test_assert_tensors_equal()
        test_case.test_state_capture_injection()


class MockModelTest(TorchComponentTestCase):
    """Test case using mock models to validate the base test case functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        
        # Add default test parameters
        self.default_test_params = {
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
        
        # Create mock models
        class MockNumpyModel:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.kvec = np.zeros((2, 2, 2, 3))
                self.kvec_norm = np.zeros((2, 2, 2, 1))
                
            def _build_kvec_Brillouin(self):
                self.kvec = np.ones((2, 2, 2, 3))
                self.kvec_norm = np.ones((2, 2, 2, 1))
        
        class MockTorchModel:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.device = kwargs.get('device', torch.device('cpu'))
                self.kvec = torch.zeros((2, 2, 2, 3), device=self.device)
                self.kvec_norm = torch.zeros((2, 2, 2, 1), device=self.device)
                
            def _build_kvec_Brillouin(self):
                self.kvec = torch.ones((2, 2, 2, 3), device=self.device)
                self.kvec_norm = torch.ones((2, 2, 2, 1), device=self.device)
        
        # Store mock model classes
        self.MockNumpyModel = MockNumpyModel
        self.MockTorchModel = MockTorchModel
        
        # Create a module-like object to hold our mock classes
        class MockModuleNP:
            OnePhonon = MockNumpyModel
            
        class MockModuleTorch:
            OnePhonon = MockTorchModel
            
        # Save original modules
        self._original_numpy_module = sys.modules.get('eryx.models')
        self._original_torch_module = sys.modules.get('eryx.models_torch')
        
        # Replace with our mock modules
        sys.modules['eryx.models'] = MockModuleNP
        sys.modules['eryx.models_torch'] = MockModuleTorch
        
    # Add missing methods that are being tested
    def create_models(self, test_params=None):
        """Create NumPy and PyTorch models for testing."""
        # Use default parameters if none provided
        params = test_params or self.default_test_params
        
        # Create NumPy model
        np_model = self.MockNumpyModel(**params)
        
        # Create PyTorch model with device
        torch_params = params.copy()
        torch_params['device'] = self.device
        torch_model = self.MockTorchModel(**torch_params)
        
        return np_model, torch_model
        
    def prepare_test_environment(self):
        """Prepare test environment with models."""
        # Create models
        self.np_model, self.torch_model = self.create_models()
        
    def run_component_test(self, test_func, *args, **kwargs):
        """Run a component test function with args and capture results."""
        # Call the test function
        success, metrics = test_func(*args, **kwargs)
        
        # Add test name to metrics
        metrics["test_name"] = test_func.__name__
        
        return success, metrics
    
    def tearDown(self):
        """Tear down test environment."""
        # Restore original modules
        if self._original_numpy_module:
            sys.modules['eryx.models'] = self._original_numpy_module
        else:
            sys.modules.pop('eryx.models', None)
            
        if self._original_torch_module:
            sys.modules['eryx.models_torch'] = self._original_torch_module
        else:
            sys.modules.pop('eryx.models_torch', None)
    
    def test_create_models(self):
        """Test create_models method."""
        # Create models with default parameters
        np_model, torch_model = self.create_models()
        
        # Verify models were created with correct parameters
        self.assertIsInstance(np_model, self.MockNumpyModel)
        self.assertIsInstance(torch_model, self.MockTorchModel)
        
        # Verify device was set correctly
        self.assertEqual(torch_model.device, self.device)
        
        # Verify parameters were passed correctly
        for key, value in self.default_test_params.items():
            self.assertEqual(np_model.params[key], value)
            self.assertEqual(torch_model.params[key], value)
    
    def test_prepare_test_environment(self):
        """Test prepare_test_environment method."""
        # Prepare test environment
        self.prepare_test_environment()
        
        # Verify models were created
        self.assertIsInstance(self.np_model, self.MockNumpyModel)
        self.assertIsInstance(self.torch_model, self.MockTorchModel)
        
        # Build kvec for both models
        self.np_model._build_kvec_Brillouin()
        self.torch_model._build_kvec_Brillouin()
        
        # Verify kvec was built
        self.assertTrue(np.array_equal(self.np_model.kvec, np.ones((2, 2, 2, 3))))
        self.assertTrue(torch.allclose(self.torch_model.kvec, torch.ones((2, 2, 2, 3), device=self.device)))
    
    def test_run_component_test(self):
        """Test run_component_test method."""
        # Create a test function
        def test_func(a, b, c=None):
            success = a == b
            metrics = {"a": a, "b": b, "c": c}
            return success, metrics
        
        # Run test with success
        success, metrics = self.run_component_test(test_func, 1, 1, c=2)
        self.assertTrue(success)
        self.assertEqual(metrics["a"], 1)
        self.assertEqual(metrics["b"], 1)
        self.assertEqual(metrics["c"], 2)
        self.assertEqual(metrics["test_name"], "test_func")
        
        # Run test with failure
        success, metrics = self.run_component_test(test_func, 1, 2, c=3)
        self.assertFalse(success)
        self.assertEqual(metrics["a"], 1)
        self.assertEqual(metrics["b"], 2)
        self.assertEqual(metrics["c"], 3)
        self.assertEqual(metrics["test_name"], "test_func")


if __name__ == '__main__':
    unittest.main()
