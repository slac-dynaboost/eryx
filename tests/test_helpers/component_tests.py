"""
Component-specific test utilities for PyTorch implementations.

This module provides test utilities for specific components of the
diffuse scattering calculations, such as k-vectors, hessians, and phonons.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional, Union, Callable

# Import tensor comparison utilities
from tests.torch_test_utils import TensorComparison


class KVectorTests:
    """Test utilities for k-vector related components."""
    
    @staticmethod
    def test_center_kvec(np_model: Any, 
                        torch_model: Any, 
                        test_cases: Optional[List[Tuple[int, int]]] = None) -> List[Dict[str, Any]]:
        """
        Test _center_kvec implementation with various inputs.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
            test_cases: List of (x, L) tuples to test, or None for defaults
        
        Returns:
            List of test results with details
        """
        # Use default test cases if none provided
        if test_cases is None:
            test_cases = [
                (0, 2), (1, 2),  # Even L
                (0, 3), (1, 3), (2, 3),  # Odd L
                (5, 10), (9, 10),  # Larger L
                (50, 100), (99, 100)  # Much larger L
            ]
        
        results = []
        
        # Call _center_kvec with each test case on both models
        for x, L in test_cases:
            np_result = np_model._center_kvec(x, L)
            torch_result = torch_model._center_kvec(x, L)
            
            # Convert torch result to Python scalar if needed
            if isinstance(torch_result, torch.Tensor):
                torch_result = torch_result.item()
            
            # Compare results
            is_equal = np.isclose(np_result, torch_result)
            
            # Store result details
            result = {
                "x": x,
                "L": L,
                "np_result": np_result,
                "torch_result": torch_result,
                "is_equal": is_equal,
                "difference": np_result - torch_result
            }
            
            results.append(result)
        
        return results
    
    @staticmethod
    def test_kvector_brillouin(np_model: Any, 
                              torch_model: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Test _build_kvec_Brillouin implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Execute _build_kvec_Brillouin on both models if not already done
        if not hasattr(np_model, 'kvec') or np_model.kvec is None:
            np_model._build_kvec_Brillouin()
            
        if not hasattr(torch_model, 'kvec') or torch_model.kvec is None:
            torch_model._build_kvec_Brillouin()
        
        # Compare resulting kvec tensors
        kvec_success, kvec_metrics = TensorComparison.compare_tensors(
            np_model.kvec, torch_model.kvec
        )
        
        # Compare resulting kvec_norm tensors
        norm_success, norm_metrics = TensorComparison.compare_tensors(
            np_model.kvec_norm, torch_model.kvec_norm
        )
        
        # Combine results
        success = kvec_success and norm_success
        metrics = {
            "kvec": kvec_metrics,
            "kvec_norm": norm_metrics,
            "success": success
        }
        
        return success, metrics
    
    @staticmethod
    def test_at_kvec_from_miller_points(np_model: Any, 
                                       torch_model: Any, 
                                       test_cases: Optional[List[Tuple[int, int, int]]] = None) -> List[Dict[str, Any]]:
        """
        Test _at_kvec_from_miller_points implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
            test_cases: List of (h, k, l) tuples to test, or None for defaults
            
        Returns:
            List of test results with details
        """
        # Use default test cases if none provided
        if test_cases is None:
            test_cases = [
                (0, 0, 0),  # Origin
                (1, 0, 0), (0, 1, 0), (0, 0, 1),  # Axes
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)  # Diagonals
            ]
            
        results = []
        
        # Call _at_kvec_from_miller_points with each test case on both models
        for hkl in test_cases:
            np_result = np_model._at_kvec_from_miller_points(hkl)
            torch_result = torch_model._at_kvec_from_miller_points(hkl)
            
            # Convert torch result to NumPy if needed
            if isinstance(torch_result, torch.Tensor):
                torch_result = torch_result.detach().cpu().numpy()
            
            # Compare results
            is_equal = np.array_equal(np_result, torch_result)
            
            # Store result details
            result = {
                "hkl": hkl,
                "np_result_shape": np_result.shape,
                "torch_result_shape": torch_result.shape,
                "is_equal": is_equal,
                "np_result_sample": np_result[:5] if len(np_result) > 0 else np_result,
                "torch_result_sample": torch_result[:5] if len(torch_result) > 0 else torch_result
            }
            
            results.append(result)
        
        return results


class HessianTests:
    """Test utilities for hessian calculation components."""
    
    @staticmethod
    def compare_hessian_structure(np_hessian: np.ndarray, 
                                 torch_hessian: torch.Tensor) -> Dict[str, Any]:
        """
        Compare structure and key properties of hessian matrices.
        
        Args:
            np_hessian: NumPy hessian array
            torch_hessian: PyTorch hessian tensor
        
        Returns:
            Dictionary with structural comparison results
        """
        # Convert torch hessian to NumPy if needed
        if isinstance(torch_hessian, torch.Tensor):
            torch_hessian_np = torch_hessian.detach().cpu().numpy()
        else:
            torch_hessian_np = torch_hessian
        
        # Compare shapes
        shape_match = np_hessian.shape == torch_hessian_np.shape
        
        # Compare value ranges
        np_min = np.min(np.abs(np_hessian))
        np_max = np.max(np.abs(np_hessian))
        torch_min = np.min(np.abs(torch_hessian_np))
        torch_max = np.max(np.abs(torch_hessian_np))
        
        # Check symmetry properties
        # For a typical hessian with shape (n_asu, n_dof, n_cell, n_asu, n_dof)
        # Check if H[i,j,0,i,j] is real and positive (diagonal elements)
        
        # Extract diagonal elements for reference cell (cell_id=0)
        if len(np_hessian.shape) == 5:
            n_asu, n_dof, _, _, _ = np_hessian.shape
            np_diag = np.array([np_hessian[i, j, 0, i, j] for i in range(n_asu) for j in range(n_dof)])
            torch_diag = np.array([torch_hessian_np[i, j, 0, i, j] for i in range(n_asu) for j in range(n_dof)])
            
            diag_real = np.allclose(np.imag(np_diag), 0) and np.allclose(np.imag(torch_diag), 0)
            diag_positive = np.all(np.real(np_diag) > 0) and np.all(np.real(torch_diag) > 0)
        else:
            diag_real = None
            diag_positive = None
        
        # Return detailed metrics
        return {
            "shape_match": shape_match,
            "np_shape": np_hessian.shape,
            "torch_shape": torch_hessian_np.shape,
            "np_min_abs": np_min,
            "np_max_abs": np_max,
            "torch_min_abs": torch_min,
            "torch_max_abs": torch_max,
            "value_range_similar": np.isclose(np_min, torch_min, rtol=1e-2) and np.isclose(np_max, torch_max, rtol=1e-2),
            "diag_real": diag_real,
            "diag_positive": diag_positive
        }
    
    @staticmethod
    def test_compute_hessian(np_model: Any, 
                            torch_model: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Test compute_hessian implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Execute compute_hessian on both models
        np_hessian = np_model.compute_hessian()
        torch_hessian = torch_model.compute_hessian()
        
        # Compare structure
        structure_metrics = HessianTests.compare_hessian_structure(np_hessian, torch_hessian)
        
        # Compare values
        success, value_metrics = TensorComparison.compare_tensors(
            np_hessian, torch_hessian, rtol=1e-4, atol=1e-6
        )
        
        # Combine results
        metrics = {
            "structure": structure_metrics,
            "values": value_metrics,
            "success": success and structure_metrics["shape_match"]
        }
        
        return metrics["success"], metrics
    
    @staticmethod
    def test_compute_gnm_K(np_model: Any, 
                          torch_model: Any, 
                          kvec: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Test compute_gnm_K implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
            kvec: Optional k-vector to use, or None for zero vector
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Get hessian from both models
        np_hessian = np_model.compute_hessian()
        torch_hessian = torch_model.compute_hessian()
        
        # Convert kvec if provided
        np_kvec = None
        torch_kvec = None
        
        if kvec is not None:
            if isinstance(kvec, torch.Tensor):
                np_kvec = kvec.detach().cpu().numpy()
                torch_kvec = kvec
            else:
                np_kvec = kvec
                torch_kvec = torch.tensor(kvec, device=torch_model.device)
        
        # Execute compute_gnm_K on both models
        np_K = np_model.gnm.compute_K(np_hessian, kvec=np_kvec)
        torch_K = torch_model.compute_gnm_K(torch_hessian, kvec=torch_kvec)
        
        # Compare values
        success, metrics = TensorComparison.compare_tensors(
            np_K, torch_K, rtol=1e-4, atol=1e-6
        )
        
        return success, metrics
    
    @staticmethod
    def test_compute_gnm_Kinv(np_model: Any, 
                             torch_model: Any, 
                             kvec: Optional[Union[np.ndarray, torch.Tensor]] = None,
                             reshape: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Test compute_gnm_Kinv implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
            kvec: Optional k-vector to use, or None for zero vector
            reshape: Whether to reshape the result
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Get hessian from both models
        np_hessian = np_model.compute_hessian()
        torch_hessian = torch_model.compute_hessian()
        
        # Convert kvec if provided
        np_kvec = None
        torch_kvec = None
        
        if kvec is not None:
            if isinstance(kvec, torch.Tensor):
                np_kvec = kvec.detach().cpu().numpy()
                torch_kvec = kvec
            else:
                np_kvec = kvec
                torch_kvec = torch.tensor(kvec, device=torch_model.device)
        
        # Execute compute_gnm_Kinv on both models
        np_Kinv = np_model.gnm.compute_Kinv(np_hessian, kvec=np_kvec, reshape=reshape)
        torch_Kinv = torch_model.compute_gnm_Kinv(torch_hessian, kvec=torch_kvec, reshape=reshape)
        
        # Compare values
        success, metrics = TensorComparison.compare_tensors(
            np_Kinv, torch_Kinv, rtol=1e-4, atol=1e-6
        )
        
        return success, metrics


class PhononTests:
    """Test utilities for phonon calculation components."""
    
    @staticmethod
    def compare_eigenvalues(np_eigenvals: np.ndarray, 
                           torch_eigenvals: torch.Tensor, 
                           rtol: float = 1e-5, 
                           atol: float = 1e-8) -> Dict[str, Any]:
        """
        Compare eigenvalues with specific handling for NaN values.
        
        Args:
            np_eigenvals: NumPy eigenvalues array
            torch_eigenvals: PyTorch eigenvalues tensor
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        
        Returns:
            Dictionary with comparison results
        """
        # Convert torch eigenvalues to NumPy if needed
        if isinstance(torch_eigenvals, torch.Tensor):
            torch_eigenvals_np = torch_eigenvals.detach().cpu().numpy()
        else:
            torch_eigenvals_np = torch_eigenvals
        
        # Check NaN patterns match
        np_nans = np.isnan(np_eigenvals)
        torch_nans = np.isnan(torch_eigenvals_np)
        nan_patterns_match = np.array_equal(np_nans, torch_nans)
        
        # Compare non-NaN values with tolerance
        if np.any(~np_nans):
            non_nan_values_match = np.allclose(
                np_eigenvals[~np_nans], 
                torch_eigenvals_np[~np_nans], 
                rtol=rtol, 
                atol=atol
            )
            
            # Calculate max absolute difference for non-NaN values
            max_abs_diff = np.max(np.abs(np_eigenvals[~np_nans] - torch_eigenvals_np[~np_nans]))
        else:
            non_nan_values_match = True
            max_abs_diff = 0.0
        
        # Return detailed results
        return {
            "success": nan_patterns_match and non_nan_values_match,
            "nan_patterns_match": nan_patterns_match,
            "non_nan_values_match": non_nan_values_match,
            "np_nan_count": np.sum(np_nans),
            "torch_nan_count": np.sum(torch_nans),
            "np_min": np.nanmin(np_eigenvals) if np.any(~np_nans) else None,
            "np_max": np.nanmax(np_eigenvals) if np.any(~np_nans) else None,
            "torch_min": np.nanmin(torch_eigenvals_np) if np.any(~torch_nans) else None,
            "torch_max": np.nanmax(torch_eigenvals_np) if np.any(~torch_nans) else None,
            "max_abs_diff": max_abs_diff
        }
    
    @staticmethod
    def compare_eigenvectors(np_eigenvecs: np.ndarray, 
                            torch_eigenvecs: torch.Tensor, 
                            rtol: float = 1e-5, 
                            atol: float = 1e-8) -> Dict[str, Any]:
        """
        Compare eigenvectors with handling for sign/phase ambiguity.
        
        Args:
            np_eigenvecs: NumPy eigenvectors array
            torch_eigenvecs: PyTorch eigenvectors tensor
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        
        Returns:
            Dictionary with comparison results
        """
        # Convert torch eigenvectors to NumPy if needed
        if isinstance(torch_eigenvecs, torch.Tensor):
            torch_eigenvecs_np = torch_eigenvecs.detach().cpu().numpy()
        else:
            torch_eigenvecs_np = torch_eigenvecs
        
        # Check shapes match
        shape_match = np_eigenvecs.shape == torch_eigenvecs_np.shape
        
        if not shape_match:
            return {
                "success": False,
                "shape_match": False,
                "np_shape": np_eigenvecs.shape,
                "torch_shape": torch_eigenvecs_np.shape
            }
        
        # For eigenvectors, we need to handle sign/phase ambiguity
        # We'll compare absolute values for real and imaginary parts
        np_abs = np.abs(np_eigenvecs)
        torch_abs = np.abs(torch_eigenvecs_np)
        
        # Compare absolute values
        abs_match = np.allclose(np_abs, torch_abs, rtol=rtol, atol=atol)
        
        # Check orthogonality
        # For each set of eigenvectors, compute V^H * V and check if it's close to identity
        if len(np_eigenvecs.shape) == 2:
            # Single set of eigenvectors
            np_ortho = np.allclose(
                np.matmul(np.conjugate(np_eigenvecs.T), np_eigenvecs),
                np.eye(np_eigenvecs.shape[1]),
                rtol=rtol, atol=atol
            )
            
            torch_ortho = np.allclose(
                np.matmul(np.conjugate(torch_eigenvecs_np.T), torch_eigenvecs_np),
                np.eye(torch_eigenvecs_np.shape[1]),
                rtol=rtol, atol=atol
            )
        else:
            # Multiple sets of eigenvectors
            np_ortho = True
            torch_ortho = True
            
            for i in range(np_eigenvecs.shape[0]):
                np_ortho = np_ortho and np.allclose(
                    np.matmul(np.conjugate(np_eigenvecs[i].T), np_eigenvecs[i]),
                    np.eye(np_eigenvecs[i].shape[1]),
                    rtol=rtol, atol=atol
                )
                
                torch_ortho = torch_ortho and np.allclose(
                    np.matmul(np.conjugate(torch_eigenvecs_np[i].T), torch_eigenvecs_np[i]),
                    np.eye(torch_eigenvecs_np[i].shape[1]),
                    rtol=rtol, atol=atol
                )
        
        # Return detailed results
        return {
            "success": shape_match and abs_match,
            "shape_match": shape_match,
            "abs_match": abs_match,
            "np_ortho": np_ortho,
            "torch_ortho": torch_ortho,
            "np_min_abs": np.min(np_abs),
            "np_max_abs": np.max(np_abs),
            "torch_min_abs": np.min(torch_abs),
            "torch_max_abs": np.max(torch_abs)
        }
    
    @staticmethod
    def test_compute_gnm_phonons(np_model: Any, 
                                torch_model: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Test compute_gnm_phonons implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Execute compute_gnm_phonons on both models if not already done
        if not hasattr(np_model, 'V') or np_model.V is None:
            np_model.compute_gnm_phonons()
            
        if not hasattr(torch_model, 'V') or torch_model.V is None:
            torch_model.compute_gnm_phonons()
        
        # Compare eigenvalues (Winv)
        eigenval_metrics = PhononTests.compare_eigenvalues(
            np_model.Winv, torch_model.Winv, rtol=1e-4, atol=1e-6
        )
        
        # Compare eigenvectors (V)
        eigenvec_metrics = PhononTests.compare_eigenvectors(
            np_model.V, torch_model.V, rtol=1e-4, atol=1e-6
        )
        
        # Combine results
        success = eigenval_metrics["success"] and eigenvec_metrics["success"]
        metrics = {
            "eigenvalues": eigenval_metrics,
            "eigenvectors": eigenvec_metrics,
            "success": success
        }
        
        return success, metrics
    
    @staticmethod
    def test_compute_covariance_matrix(np_model: Any, 
                                      torch_model: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Test compute_covariance_matrix implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Execute compute_covariance_matrix on both models if not already done
        if not hasattr(np_model, 'covar') or np_model.covar is None:
            np_model.compute_covariance_matrix()
            
        if not hasattr(torch_model, 'covar') or torch_model.covar is None:
            torch_model.compute_covariance_matrix()
        
        # Compare covariance matrices
        covar_success, covar_metrics = TensorComparison.compare_tensors(
            np_model.covar, torch_model.covar, rtol=1e-4, atol=1e-6
        )
        
        # Compare ADPs
        adp_success, adp_metrics = TensorComparison.compare_tensors(
            np_model.ADP, torch_model.ADP, rtol=1e-4, atol=1e-6
        )
        
        # Combine results
        success = covar_success and adp_success
        metrics = {
            "covar": covar_metrics,
            "adp": adp_metrics,
            "success": success
        }
        
        return success, metrics


class DisorderTests:
    """Test utilities for disorder calculation components."""
    
    @staticmethod
    def test_apply_disorder(np_model: Any, 
                           torch_model: Any, 
                           rank: int = -1, 
                           use_data_adp: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Test apply_disorder implementation against NumPy version.
        
        Args:
            np_model: NumPy model instance
            torch_model: PyTorch model instance
            rank: Phonon mode rank to use (-1 for all modes)
            use_data_adp: Whether to use data ADPs
        
        Returns:
            Tuple of (bool success, dict metrics) with comparison details
        """
        # Execute apply_disorder on both models
        np_result = np_model.apply_disorder(rank=rank, use_data_adp=use_data_adp)
        torch_result = torch_model.apply_disorder(rank=rank, use_data_adp=use_data_adp)
        
        # Compare results
        success, metrics = TensorComparison.compare_tensors(
            np_result, torch_result, rtol=1e-4, atol=1e-6, equal_nan=True
        )
        
        # Add additional metrics
        metrics["rank"] = rank
        metrics["use_data_adp"] = use_data_adp
        
        # Calculate correlation coefficient for non-NaN values
        np_flat = np_result.flatten()
        torch_flat = torch_result.detach().cpu().numpy().flatten()
        
        mask = ~np.isnan(np_flat) & ~np.isnan(torch_flat)
        if np.any(mask):
            correlation = np.corrcoef(np_flat[mask], torch_flat[mask])[0, 1]
            metrics["correlation"] = correlation
        
        return success, metrics
