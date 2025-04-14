"""
Tests for arbitrary q-vector input support in the OnePhonon model.

This module contains tests to verify the functionality of arbitrary q-vector
inputs in the PyTorch implementation of diffuse scattering calculations.
"""

import os
import unittest
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from tests.test_base import TestBase
from eryx.models_torch import OnePhonon
try:
    from tests.torch_test_utils import TensorComparison
except ImportError:
    TensorComparison = None



class TestArbitraryQVectors(TestBase):
    """Test case for arbitrary q-vector input support."""
    ATTRIBUTES_WITH_SHAPE_DIFF = {'kvec', 'kvec_norm', 'V', 'Winv'}
    
    def setUp(self):
        """Set up test environment."""
        # Call parent setUp
        super().setUp()
        
        # Set module name for log paths
        self.module_name = "eryx.models_torch"
        self.class_name = "OnePhonon"
        
        # Default test parameters
        self.pdb_path = 'tests/pdbs/5zck_p1.pdb'
        
        # Grid parameters for comparison
        self.grid_params = {
            'hsampling': [-2, 2, 3],
            'ksampling': [-2, 2, 3],
            'lsampling': [-2, 2, 3],
            'expand_p1': True,
            'res_limit': 0.0
        }
        
        # Use a small, consistent grid for initial testing
        self.hsampling = [-2, 2, 2]  # (min, max, steps_per_miller_index)
        self.ksampling = [-2, 2, 2]
        self.lsampling = [-2, 2, 2]
        
        # Common parameters for OnePhonon initialization
        self.common_params = {
            'expand_p1': True,
            'group_by': 'asu',
            'res_limit': 0.0,
            'model': 'gnm',
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0,
            'n_processes': 1  # Use single process for deterministic results
        }
        
        # Force CPU for deterministic comparisons in this phase
        self.device = torch.device('cpu')

        # Define standard tolerance levels based on previous debugging efforts
        self.tight_tol = {'rtol': 1e-12, 'atol': 1e-14}  # For exact matches (grids, etc.)
        self.medium_tol = {'rtol': 1e-6, 'atol': 1e-8}   # For standard calculations
        self.loose_tol = {'rtol': 1e-5, 'atol': 1e-7}    # For results involving eigh/inverse
    
    # Define the execution order of key methods within OnePhonon initialization and calculation
    # This order is crucial for executing prerequisites correctly.
    METHOD_EXECUTION_ORDER = [
        '_setup',  # Grid, crystal, dimensions
        '_build_A',  # Projection matrix
        '_build_M',  # Final mass matrix and Linv (implicitly tests _build_M_allatoms, _project_M)
        '_build_kvec_Brillouin',  # k-vectors
        'compute_gnm_phonons',  # Eigenvectors/values (V, Winv) (implicitly tests compute_hessian)
        'compute_covariance_matrix',  # Covariance matrix (covar, ADP) (implicitly tests compute_hessian)
        'apply_disorder',  # Final Intensity (Id)
    ]

    # Map methods to the primary attributes they compute/modify or None if they return the value
    METHOD_OUTPUT_ATTRIBUTES = {
        '_setup': ['hkl_grid', 'q_grid', 'map_shape', 'res_mask', 'crystal', 'n_asu', 'n_atoms_per_asu', 'n_dof_per_asu', 'n_dof_per_asu_actual'],
        '_build_A': ['Amat'],
        '_build_M': ['Linv'],
        '_build_kvec_Brillouin': ['kvec', 'kvec_norm'],
        'compute_gnm_phonons': ['V', 'Winv'],
        'compute_covariance_matrix': ['covar', 'ADP'],
        'apply_disorder': None  # Indicates this method returns the value directly
    }
    
    def _get_comparison_tolerances(self, method_name: str) -> Dict[str, float]:
        """Return appropriate tolerances based on the method being tested."""
        # Use tolerances defined in setUp, based on conventions from grid_debug.md
        if method_name in ['_setup', '_build_A', '_build_M_allatoms', '_project_M', '_build_kvec_Brillouin']:
            # These should match very closely (potentially bit-for-bit if NumPy grid is used)
            return self.tight_tol
        elif method_name in ['_build_M', 'compute_hessian']:  # Linv involves inversion/SVD
            return self.medium_tol
        elif method_name in ['compute_gnm_phonons', 'compute_covariance_matrix', 'apply_disorder']:
            # These involve eigendecomposition or complex summation, allow looser tolerance
            return self.loose_tol
        else:
            # Default for unknown methods
            return self.medium_tol
    
    def _compare_outputs(self, method_name: str, output_grid: Any, output_q: Any) -> bool:
        """Compares outputs using appropriate tolerances and methods."""
        tolerances = self._get_comparison_tolerances(method_name)
        print(f"Comparing results for '{method_name}' using tolerances: {tolerances}")

        # Special comparison for eigenvectors (V) in compute_gnm_phonons
        if method_name == 'compute_gnm_phonons':
            # Fetch V and Winv attributes from the model instances
            grid_V = getattr(output_grid, 'V', None)
            grid_Winv = getattr(output_grid, 'Winv', None)
            q_V = getattr(output_q, 'V', None)
            q_Winv = getattr(output_q, 'Winv', None)

            if grid_V is None or q_V is None or grid_Winv is None or q_Winv is None:
                print(f"Error: Could not retrieve V or Winv attributes for {method_name}")
                return False

            # 1. Compare Winv (eigenvalues) directly, handling NaNs
            try:
                print(f"  Comparing Winv (shape: {grid_Winv.shape})...")
                # Use TensorComparison if available, otherwise fallback
                if TensorComparison:
                    TensorComparison.assert_tensors_equal(grid_Winv, q_Winv, equal_nan=True, **tolerances)
                else:
                    np.testing.assert_allclose(
                        grid_Winv.detach().cpu().numpy(),
                        q_Winv.detach().cpu().numpy(),
                        equal_nan=True, **tolerances
                    )
                print("  Winv comparison successful.")
            except AssertionError as e:
                print(f"  Winv comparison FAILED: {e}")
                return False  # Fail early if eigenvalues don't match

            # 2. Compare V (eigenvectors) using projection matrices P = V @ V.H
            try:
                print(f"  Comparing V (shape: {grid_V.shape}) via projection matrix...")
                grid_P = grid_V @ grid_V.conj().transpose(-1, -2)
                q_P = q_V @ q_V.conj().transpose(-1, -2)
                if TensorComparison:
                    TensorComparison.assert_tensors_equal(grid_P, q_P, equal_nan=True, **tolerances)
                else:
                    np.testing.assert_allclose(
                        grid_P.detach().cpu().numpy(),
                        q_P.detach().cpu().numpy(),
                        equal_nan=True, **tolerances
                    )
                print("  V (projection matrix) comparison successful.")
                return True  # Both Winv and V (projection) matched
            except AssertionError as e:
                print(f"  V (projection matrix) comparison FAILED: {e}")
                return False
        else:
            # --- General Comparison Logic for other methods/attributes ---
            # Check if both outputs are tensors for shape comparison
            is_grid_tensor = isinstance(output_grid, torch.Tensor)
            is_q_tensor = isinstance(output_q, torch.Tensor)

            # Check shapes *before* converting, only if both are tensors
            shapes_match = True
            if is_grid_tensor and is_q_tensor:
                if output_grid.shape != output_q.shape:
                    shapes_match = False
                    # Extract attribute name if comparing attributes
                    attr_name = None
                    if "attribute: " in method_name: # Check if method_name indicates an attribute comparison
                        try:
                            attr_name = method_name.split("attribute: ")[1].split(')')[0]
                        except IndexError:
                            pass # Keep attr_name as None if parsing fails

                    # Check if this attribute is expected to have different shapes
                    if attr_name and attr_name in TestArbitraryQVectors.ATTRIBUTES_WITH_SHAPE_DIFF:
                        print(f"  INFO: Shapes differ for attribute '{attr_name}' as expected ({output_grid.shape} vs {output_q.shape}). Skipping numerical comparison.")
                        return True # Treat expected shape difference as success for this check
                    else:
                        print(f"  ERROR: Unexpected shape mismatch for '{method_name}'. Grid: {output_grid.shape}, Q: {output_q.shape}")
                        # Fall through to numpy conversion and assertion, which will fail clearly
            elif type(output_grid) != type(output_q):
                # Handle cases where one is tensor and other is not, or types differ
                print(f"  ERROR: Type mismatch for '{method_name}'. Grid: {type(output_grid)}, Q: {type(output_q)}")
                shapes_match = False # Treat type mismatch effectively as shape mismatch

            # Convert tensors to NumPy for comparison using allclose
            if is_grid_tensor:
                output_grid_np = output_grid.detach().cpu().numpy()
            elif isinstance(output_grid, (np.ndarray, int, float, bool, tuple, list)):
                output_grid_np = np.array(output_grid)  # Handle non-tensor outputs if needed
            else:
                print(f"  Unsupported type for grid output: {type(output_grid)}")
                return False

            if is_q_tensor:
                output_q_np = output_q.detach().cpu().numpy()
            elif isinstance(output_q, (np.ndarray, int, float, bool, tuple, list)):
                output_q_np = np.array(output_q)
            else:
                print(f"  Unsupported type for q output: {type(output_q)}")
                return False

            # If shapes didn't match for an *unexpected* attribute, the assertion below will fail clearly.
            if not shapes_match:
                print(f"  Comparison FAILED for '{method_name}' due to unexpected shape/type mismatch.")
                return False

            # Perform comparison
            try:
                print(f"  Grid Shape: {output_grid_np.shape}, Q Shape: {output_q_np.shape}")
                print(f"  Grid dtype: {output_grid_np.dtype}, Q dtype: {output_q_np.dtype}")
                np.testing.assert_allclose(output_grid_np, output_q_np, equal_nan=True, **tolerances)
                print(f"  Comparison successful for '{method_name}'.")
                return True
            except AssertionError as e:
                # Calculate max difference for better debugging info
                try:
                    abs_diff = np.abs(output_grid_np - output_q_np)
                    max_diff = np.nanmax(abs_diff)
                    print(f"  Comparison FAILED for '{method_name}'. Max difference: {max_diff}\n  Details: {e}")
                except (TypeError, ValueError):  # Handle non-numeric types or further shape issues during subtraction
                    print(f"  Comparison FAILED for '{method_name}'. Error during difference calculation.\n  Details: {e}")
                return False
    
    def run_method_equivalence_test(self, target_method_name: str):
        """
        Core test harness to compare a method's output between grid and arbitrary-q modes.

        Args:
            target_method_name: The name (string) of the OnePhonon method to test.
        """
        print(f"\n===== Testing Method Equivalence for: {target_method_name} =====")

        # --- 1. Initialization ---
        print("Initializing grid-based model...")
        # Ensure high precision is used during initialization by passing dtype arguments if possible,
        # or by setting them after initialization. For now, we set after.
        model_grid = OnePhonon(
            pdb_path=self.pdb_path,
            hsampling=self.hsampling, ksampling=self.ksampling, lsampling=self.lsampling,
            device=self.device, **self.common_params
        )
        model_grid.real_dtype = torch.float64
        model_grid.complex_dtype = torch.complex128
        # Re-run relevant setup steps if dtypes changed internal tensors
        # For simplicity in Phase 1, we assume __init__ handles precision correctly based on internal logic.
        # If tests fail later due to precision, revisit __init__ or add re-setup steps here.
        print(f"Grid model initialized. q_grid shape: {model_grid.q_grid.shape}, dtype: {model_grid.q_grid.dtype}")

        print("Extracting q-grid from grid model...")
        # Ensure the extracted q_grid has the correct high precision
        q_vectors_from_grid = model_grid.q_grid.clone().detach().to(dtype=torch.float64)

        print("Initializing arbitrary q-vector model...")
        model_q = OnePhonon(
            pdb_path=self.pdb_path,
            q_vectors=q_vectors_from_grid,  # Use the extracted q-grid
            device=self.device, **self.common_params
        )
        model_q.real_dtype = torch.float64
        model_q.complex_dtype = torch.complex128
        # Re-run relevant setup steps if needed
        print(f"Arbitrary-q model initialized. q_grid shape: {model_q.q_grid.shape}, dtype: {model_q.q_grid.dtype}")

        # Initial check: q_grids should be identical
        try:
            print("Verifying initial q_grid match...")
            # Use TensorComparison if available, otherwise fallback
            if TensorComparison:
                TensorComparison.assert_tensors_equal(
                    model_grid.q_grid, model_q.q_grid, **self.tight_tol,
                    msg="Initial q_grid tensors do not match between models"
                )
            else:
                np.testing.assert_allclose(
                    model_grid.q_grid.detach().cpu().numpy(),
                    model_q.q_grid.detach().cpu().numpy(),
                    **self.tight_tol
                )
            print("Initial q_grid tensors match.")
        except AssertionError as e:
            self.fail(f"Prerequisite check failed: Initial q_grids differ significantly. {e}")


        # --- 2. Execute Prerequisites ---
        print("Ensuring prerequisites are met for target method...")
        # __init__ already calls _setup, _build_A, _build_M, _build_kvec_Brillouin via _setup_phonons

        # Define prerequisites for later methods
        prerequisites = {
            'compute_covariance_matrix': ['compute_gnm_phonons'],
            'apply_disorder': ['compute_gnm_phonons', 'compute_covariance_matrix']
            # Add more if other methods have specific runtime dependencies not covered by __init__
        }

        # Determine methods to run before the target
        methods_to_run = []
        try:
            target_index = self.METHOD_EXECUTION_ORDER.index(target_method_name)
            # Collect all methods from the beginning up to *before* the target
            required_by_order = self.METHOD_EXECUTION_ORDER[:target_index]

            # Check explicit prerequisites for the target method
            explicit_prereqs = prerequisites.get(target_method_name, [])

            # Combine and ensure unique methods are run in the correct order
            # We only need to explicitly call methods *not* already run by __init__
            # or by prerequisites of earlier methods in the chain.
            # For GNM model, __init__ runs up to compute_covariance_matrix if model='gnm'.
            # Let's explicitly call compute_gnm_phonons and compute_covariance_matrix
            # if they appear before the target method in the execution order.

            methods_to_explicitly_run = []
            if 'compute_gnm_phonons' in required_by_order:
                methods_to_explicitly_run.append('compute_gnm_phonons')
            if 'compute_covariance_matrix' in required_by_order:
                # Ensure compute_gnm_phonons is also run if needed
                if 'compute_gnm_phonons' not in methods_to_explicitly_run:
                    methods_to_explicitly_run.append('compute_gnm_phonons')
                methods_to_explicitly_run.append('compute_covariance_matrix')

            print(f"  Methods to explicitly run before '{target_method_name}': {methods_to_explicitly_run}")

            # Execute these methods on both models
            for method_name in methods_to_explicitly_run:
                print(f"  Running prerequisite: {method_name}...")
                if not hasattr(model_grid, method_name) or not hasattr(model_q, method_name):
                    print(f"    Skipping {method_name} - not found on one or both models.")
                    continue
                try:
                    print(f"    Executing {method_name} on model_grid...")
                    getattr(model_grid, method_name)()
                    print(f"    Executing {method_name} on model_q...")
                    getattr(model_q, method_name)()
                    print(f"    {method_name} executed successfully on both models.")
                except Exception as e:
                    self.fail(f"Error executing prerequisite method '{method_name}': {e}")

        except ValueError:
            self.fail(f"Target method '{target_method_name}' not found in defined METHOD_EXECUTION_ORDER.")
        except Exception as e:
            self.fail(f"Error during prerequisite execution setup: {e}")

        # --- Prerequisite execution complete ---

        # --- 3. Execute Target Method ---
        print(f"Executing target method: {target_method_name}...")
        if not hasattr(model_grid, target_method_name) or not hasattr(model_q, target_method_name):
            self.fail(f"Target method '{target_method_name}' not found on one or both models.")

        output_grid = None
        output_q = None
        try:
            method_grid = getattr(model_grid, target_method_name)
            method_q = getattr(model_q, target_method_name)

            print(f"  Calling {target_method_name} on model_grid...")
            output_grid = method_grid()
            print(f"  Calling {target_method_name} on model_q...")
            output_q = method_q()
            print(f"{target_method_name} executed on both models.")

        except Exception as e:
            self.fail(f"Error executing target method '{target_method_name}': {e}")

        # --- 4. Compare Outputs / Attributes ---
        print(f"Comparing results for {target_method_name}...")
        # Use the class constant METHOD_OUTPUT_ATTRIBUTES
        attributes_to_compare = self.METHOD_OUTPUT_ATTRIBUTES.get(target_method_name, [])

        comparison_passed = True
        if attributes_to_compare is None:
            # Method returns the value directly - compare return values
            print(f"  Comparing direct return values of {target_method_name}...")
            if not self._compare_outputs(target_method_name, output_grid, output_q):
                comparison_passed = False
        elif isinstance(attributes_to_compare, list):
            # Method modifies attributes on self - compare listed attributes
            if not attributes_to_compare:
                print(f"  Warning: No output attributes defined for {target_method_name}. Assuming success if execution finished.")
            else:
                print(f"  Comparing attributes: {attributes_to_compare}")
                for attr in attributes_to_compare:
                    # Handle nested attributes like 'gnm.hessian' safely
                    def safe_getattr(obj, attr_path, default=None):
                        parts = attr_path.split('.')
                        current = obj
                        for part in parts:
                            # Check if current object is dict-like or has attribute
                            is_dict_like = isinstance(current, dict)
                            has_the_attr = hasattr(current, part)
                            if not (is_dict_like and part in current) and not (not is_dict_like and has_the_attr):
                                return default
                            current = current[part] if is_dict_like else getattr(current, part)
                        return current

                    val_grid = safe_getattr(model_grid, attr)
                    val_q = safe_getattr(model_q, attr)

                    if val_grid is None and val_q is None:
                        print(f"    Attribute '{attr}' is None in both models. Skipping comparison.")
                        continue
                    # Allow comparison even if one is None, _compare_outputs handles it

                    print(f"    Comparing attribute: {attr}")
                    # Call compare_outputs, which now handles shape checks internally
                    if not self._compare_outputs(f"{target_method_name} (attribute: {attr})", val_grid, val_q):
                        comparison_passed = False
                        # Optionally break on first failure or collect all failures
                        break
        else:
            print(f"  Error: Invalid definition in METHOD_OUTPUT_ATTRIBUTES for '{target_method_name}'.")
            comparison_passed = False


        self.assertTrue(comparison_passed, f"Equivalence test failed for method '{target_method_name}'. See logs above for details.")
        print(f"===== Equivalence Test PASSED for: {target_method_name} =====")
    
    def create_q_vectors_from_grid(self) -> Tuple[torch.Tensor, OnePhonon]:
        """
        Create q-vectors from grid-based approach for equivalence testing.
        
        Returns:
            Tuple containing:
                - q_vectors: Tensor of q-vectors from grid
                - grid_model: OnePhonon instance using grid-based approach
        """
        # Create grid-based model
        grid_model = OnePhonon(
            self.pdb_path,
            **self.grid_params,
            device=self.device
        )
        
        # Extract q-vectors from grid model
        q_vectors = grid_model.q_grid.clone().detach()
        
        return q_vectors, grid_model
    
    def test_constructor_validation(self):
        """Test constructor validation for q-vectors parameter."""
        # Test with invalid q-vectors type
        with self.assertRaises(ValueError):
            OnePhonon(
                self.pdb_path,
                q_vectors=np.array([[0.1, 0.2, 0.3]]),  # NumPy array instead of tensor
                device=self.device
            )
        
        # Test with invalid q-vectors shape
        with self.assertRaises(ValueError):
            OnePhonon(
                self.pdb_path,
                q_vectors=torch.tensor([0.1, 0.2, 0.3]),  # 1D tensor instead of 2D
                device=self.device
            )
        
        # Test with missing required parameters
        with self.assertRaises(ValueError):
            OnePhonon(
                self.pdb_path,
                hsampling=None,  # Missing required parameter
                ksampling=[-2, 2, 3],
                lsampling=[-2, 2, 3],
                device=self.device
            )
        
        # Test with valid q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=torch.tensor([[0.1, 0.2, 0.3]], device=self.device),
            device=self.device
        )
        
        # Verify model attributes
        self.assertTrue(model.use_arbitrary_q)
        self.assertEqual(model.q_grid.shape, (1, 3))
        self.assertTrue(model.q_grid.requires_grad)
    
    def test_grid_equivalence(self):
        """
        Test that using explicit q-vectors from a grid produces identical results
        to the grid-based approach.
        """
        # Get q-vectors from grid model
        q_vectors, grid_model = self.create_q_vectors_from_grid()
        
        # Create arbitrary q-vector model with the same q-vectors
        q_model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Compare q_grid tensors (should be identical)
        self.assertTrue(torch.allclose(grid_model.q_grid, q_model.q_grid))
        
        # Compare hkl_grid tensors (should be close within numerical precision)
        self.assertTrue(torch.allclose(grid_model.hkl_grid, q_model.hkl_grid, rtol=1e-5, atol=1e-8))
        
        # Compare resolution masks
        self.assertTrue(torch.all(grid_model.res_mask == q_model.res_mask))
    
    def test_gradient_flow(self):
        """
        Test that gradients flow correctly through arbitrary q-vector calculations.
        """
        # Create a small set of q-vectors that requires gradients
        q_vectors = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], device=self.device, requires_grad=True)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Verify q_grid has gradients enabled
        self.assertTrue(model.q_grid.requires_grad)
        
        # Create a simple computation to test gradient flow
        # Sum of all elements in q_grid
        q_sum = torch.sum(model.q_grid)
        
        # Compute backward pass
        q_sum.backward()
        
        # Verify gradients are computed
        self.assertIsNotNone(q_vectors.grad)
        self.assertTrue(torch.all(q_vectors.grad == torch.ones_like(q_vectors)))
    
    def test_custom_q_vectors(self):
        """
        Test with custom q-vectors that don't follow a grid pattern.
        """
        # Create a custom set of q-vectors
        q_vectors = torch.tensor([
            [0.123, 0.456, 0.789],
            [1.234, 2.345, 3.456],
            [-0.123, -0.456, -0.789]
        ], device=self.device)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Verify model attributes
        self.assertTrue(model.use_arbitrary_q)
        self.assertEqual(model.q_grid.shape, (3, 3))
        self.assertTrue(torch.allclose(model.q_grid, q_vectors))
        
        # Verify map_shape is set correctly
        self.assertEqual(model.map_shape, (3, 1, 1))
        
        # Verify hkl_grid is computed correctly
        # q = 2π * A_inv^T * hkl, so hkl = (1/2π) * q * (A_inv^T)^-1
        A_inv_tensor = torch.tensor(model.model.A_inv, dtype=torch.float32, device=self.device)
        scaling_factor = 1.0 / (2.0 * torch.pi)
        A_inv_T_inv = torch.inverse(A_inv_tensor.T)
        expected_hkl = torch.matmul(q_vectors * scaling_factor, A_inv_T_inv)
        
        self.assertTrue(torch.allclose(model.hkl_grid, expected_hkl, rtol=1e-5, atol=1e-8))
    
    def test_basic_initialization(self):
        """
        Test that a model with arbitrary q-vectors initializes correctly.
        """
        # Create a custom set of q-vectors
        q_vectors = torch.tensor([
            [0.123, 0.456, 0.789],
            [1.234, 2.345, 3.456],
            [-0.123, -0.456, -0.789]
        ], device=self.device)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Verify model attributes
        self.assertTrue(model.use_arbitrary_q)
        self.assertEqual(model.q_grid.shape, (3, 3))
        self.assertTrue(model.q_grid.requires_grad)
        
        # Verify tensors have correct shapes
        self.assertEqual(model.kvec.shape, (3, 3))
        self.assertEqual(model.kvec_norm.shape, (3, 1))
        self.assertTrue(model.kvec.requires_grad)
        self.assertTrue(model.kvec_norm.requires_grad)
        
        # Verify V and Winv tensors exist and have requires_grad
        self.assertTrue(hasattr(model, 'V'))
        self.assertTrue(hasattr(model, 'Winv'))
        self.assertTrue(model.V.requires_grad)
        self.assertTrue(model.Winv.requires_grad)
    
    def test_build_kvec_brillouin(self):
        """
        Test _build_kvec_Brillouin method with arbitrary q-vectors.
        """
        # Create a custom set of q-vectors
        q_vectors = torch.tensor([
            [0.123, 0.456, 0.789],
            [1.234, 2.345, 3.456],
            [-0.123, -0.456, -0.789]
        ], device=self.device, requires_grad=True)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Call _build_kvec_Brillouin explicitly
        model._build_kvec_Brillouin()
        
        # Verify kvec and kvec_norm tensor shapes
        self.assertEqual(model.kvec.shape, (3, 3))
        self.assertEqual(model.kvec_norm.shape, (3, 1))
        
        # Verify kvec = q_grid/(2π)
        expected_kvec = q_vectors / (2.0 * torch.pi)
        self.assertTrue(torch.allclose(model.kvec, expected_kvec, rtol=1e-5, atol=1e-8))
        
        # Verify both tensors have requires_grad=True
        self.assertTrue(model.kvec.requires_grad)
        self.assertTrue(model.kvec_norm.requires_grad)
        
        # Test gradient flow
        # Create a simple loss function
        loss = torch.sum(model.kvec)
        
        # Compute backward pass
        loss.backward()
        
        # Verify gradients are computed
        self.assertIsNotNone(q_vectors.grad)
        self.assertTrue(torch.all(q_vectors.grad > 0))
    
    def test_at_kvec_from_miller_points(self):
        """
        Test _at_kvec_from_miller_points method with arbitrary q-vectors.
        """
        # Create a custom set of q-vectors
        q_vectors = torch.tensor([
            [0.123, 0.456, 0.789],
            [1.234, 2.345, 3.456],
            [-0.123, -0.456, -0.789]
        ], device=self.device)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Test with direct index input
        direct_idx = 1
        result = model._at_kvec_from_miller_points(direct_idx)
        self.assertEqual(result, direct_idx)
        
        # Test with tensor of indices
        indices_tensor = torch.tensor([0, 2], device=self.device)
        result = model._at_kvec_from_miller_points(indices_tensor)
        self.assertTrue(torch.all(result == indices_tensor))
        
        # Test with Miller indices tuple
        # Create a q-vector that should be close to one in our list
        A_inv_tensor = torch.tensor(model.model.A_inv, dtype=torch.float32, device=self.device)
        hkl = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        target_q = 2 * torch.pi * torch.matmul(A_inv_tensor.T, hkl)
        
        # Find the closest q-vector in our list
        distances = torch.norm(q_vectors - target_q, dim=1)
        expected_idx = torch.argmin(distances).item()
        
        # Test the method with the same hkl
        result = model._at_kvec_from_miller_points((1.0, 2.0, 3.0))
        self.assertEqual(result, expected_idx)
    
    def test_shape_handling_methods(self):
        """
        Test shape handling methods with arbitrary q-vectors.
        """
        # Create a custom set of q-vectors
        q_vectors = torch.tensor([
            [0.123, 0.456, 0.789],
            [1.234, 2.345, 3.456],
            [-0.123, -0.456, -0.789]
        ], device=self.device)
        
        # Create model with these q-vectors
        model = OnePhonon(
            self.pdb_path,
            q_vectors=q_vectors,
            device=self.device
        )
        
        # Create test tensors
        test_tensor_2d = torch.rand((3, 5), device=self.device)
        test_tensor_3d = torch.rand((3, 3, 3), device=self.device)
        
        # Test to_batched_shape (should be identity operation)
        result_2d = model.to_batched_shape(test_tensor_2d)
        self.assertTrue(torch.all(result_2d == test_tensor_2d))
        self.assertEqual(result_2d.shape, test_tensor_2d.shape)
        
        result_3d = model.to_batched_shape(test_tensor_3d)
        self.assertTrue(torch.all(result_3d == test_tensor_3d))
        self.assertEqual(result_3d.shape, test_tensor_3d.shape)
        
        # Test to_original_shape (should be identity operation)
        result_2d = model.to_original_shape(test_tensor_2d)
        self.assertTrue(torch.all(result_2d == test_tensor_2d))
        self.assertEqual(result_2d.shape, test_tensor_2d.shape)
        
        result_3d = model.to_original_shape(test_tensor_3d)
        self.assertTrue(torch.all(result_3d == test_tensor_3d))
        self.assertEqual(result_3d.shape, test_tensor_3d.shape)
    
    def test_Amat_equivalence(self):
        """Test equivalence of the Amat calculation (_build_A)."""
        self.run_method_equivalence_test('_build_A')
        
    def test_Linv_equivalence(self):
        """Test equivalence of the Linv calculation (_build_M)."""
        self.run_method_equivalence_test('_build_M')
        
    def test_kvec_brillouin_equivalence(self):
        """Test equivalence of _build_kvec_Brillouin method."""
        self.run_method_equivalence_test('_build_kvec_Brillouin')
        
    def test_apply_disorder_equivalence(self):
        """Test equivalence of the final apply_disorder method."""
        self.run_method_equivalence_test('apply_disorder')

if __name__ == '__main__':
    unittest.main()
