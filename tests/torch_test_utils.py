"""
Utilities for testing PyTorch implementations against NumPy ground truth.

This module provides utilities for comparing PyTorch tensors with NumPy arrays,
capturing and injecting model state, and other testing helpers.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional, Union


class TensorComparison:
    """
    Utilities for comparing PyTorch tensors with NumPy arrays.
    
    This class provides methods for comparing PyTorch tensors with NumPy arrays
    with detailed metrics and error reporting.
    """
    
    @staticmethod
    def compare_tensors(np_array: np.ndarray, 
                        torch_tensor: torch.Tensor, 
                        rtol: float = 1e-5, 
                        atol: float = 1e-8, 
                        check_nans: bool = True, 
                        equal_nan: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare NumPy array with PyTorch tensor with detailed metrics.
        
        Args:
            np_array: NumPy array to compare
            torch_tensor: PyTorch tensor to compare
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            check_nans: Whether to verify NaN patterns match
            equal_nan: Whether to consider NaN values as equal
            
        Returns:
            Tuple of (bool success, dict metrics) with detailed comparison results
        """
        # Convert tensor to NumPy if needed
        if torch_tensor is None and np_array is None:
            return True, {"message": "Both inputs are None"}
        
        if torch_tensor is None or np_array is None:
            return False, {"message": f"One input is None: np_array={np_array is not None}, torch_tensor={torch_tensor is not None}"}
        
        # Convert tensor to NumPy *before* any NumPy operations
        if isinstance(torch_tensor, torch.Tensor):
            # Detach from graph, move to CPU, convert to NumPy
            torch_array = torch_tensor.detach().cpu().numpy()
        else:
            # If input is already NumPy or other compatible type, use it directly
            torch_array = torch_tensor
            
        # Check shapes match
        if np_array.shape != torch_array.shape:
            return False, {
                "message": "Shape mismatch",
                "np_shape": np_array.shape,
                "torch_shape": torch_array.shape
            }
            
        # Handle empty arrays
        if np_array.size == 0:
            return True, {"message": "Empty arrays"}
            
        # Handle NaN values
        np_nans = np.isnan(np_array)
        torch_nans = np.isnan(torch_array)
        
        if check_nans and not np.array_equal(np_nans, torch_nans):
            nan_mismatch_count = np.sum(np_nans != torch_nans)
            return False, {
                "message": "NaN pattern mismatch",
                "nan_mismatch_count": nan_mismatch_count,
                "np_nan_count": np.sum(np_nans),
                "torch_nan_count": np.sum(torch_nans)
            }
            
        # Mask out NaNs for comparison if equal_nan is True
        if equal_nan:
            mask = ~(np_nans | torch_nans)
            if not np.any(mask):
                return True, {"message": "All values are NaN"}
            np_masked = np_array[mask]
            torch_masked = torch_array[mask]
        else:
            # When not using equal_nan, we still need to handle NaNs properly
            # for the comparison to work correctly
            if np.any(np_nans) and check_nans:
                # If we've reached here, NaN patterns match, so we can mask them out
                mask = ~np_nans
                if not np.any(mask):
                    return True, {"message": "All values are NaN"}
                np_masked = np_array[mask]
                torch_masked = torch_array[mask]
            else:
                np_masked = np_array
                torch_masked = torch_array
            
        # Calculate metrics
        abs_diff = np.abs(np_masked - torch_masked)
        max_abs_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
        mean_abs_diff = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0
            
        # Calculate relative difference for non-zero values
        non_zero_mask = np_masked != 0
        if np.any(non_zero_mask):
            rel_diff = abs_diff[non_zero_mask] / np.abs(np_masked[non_zero_mask])
            max_rel_diff = float(np.max(rel_diff))
            mean_rel_diff = float(np.mean(rel_diff))
        else:
            max_rel_diff = 0.0
            mean_rel_diff = 0.0
            
        # Check if arrays are close
        try:
            is_close = np.allclose(np_masked, torch_masked, rtol=rtol, atol=atol)
        except:
            # Handle case where comparison might fail due to type issues
            is_close = np.allclose(np_masked.astype(np.float32), 
                                  torch_masked.astype(np.float32), 
                                  rtol=rtol, atol=atol)
        
        # Prepare detailed metrics
        metrics = {
            "success": is_close,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": mean_rel_diff,
            "np_min": np.min(np_masked) if np_masked.size > 0 else None,
            "np_max": np.max(np_masked) if np_masked.size > 0 else None,
            "torch_min": np.min(torch_masked) if torch_masked.size > 0 else None,
            "torch_max": np.max(torch_masked) if torch_masked.size > 0 else None
        }
        
        return is_close, metrics
    
    @staticmethod
    def assert_tensors_equal(np_array: np.ndarray, 
                            torch_tensor: torch.Tensor, 
                            rtol: float = 1e-5, 
                            atol: float = 1e-8, 
                            check_nans: bool = True, 
                            equal_nan: bool = False, 
                            msg: Optional[str] = None) -> None:
        """
        Assert that NumPy array and PyTorch tensor are equivalent with detailed error reporting.
        
        Args:
            np_array: NumPy array to compare
            torch_tensor: PyTorch tensor to compare
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            check_nans: Whether to verify NaN patterns match
            equal_nan: Whether to consider NaN values as equal
            msg: Optional error message prefix
        """
        # Use compare_tensors to get detailed metrics
        success, metrics = TensorComparison.compare_tensors(
            np_array, torch_tensor, rtol, atol, check_nans, equal_nan
        )
        
        if not success:
            # Create clear error message if comparison fails
            error_msg = msg or "Tensor comparison failed"
            
            if "message" in metrics:
                error_msg += f": {metrics['message']}"
                
            if "np_shape" in metrics and "torch_shape" in metrics:
                error_msg += f"\nShape mismatch: NumPy {metrics['np_shape']} vs PyTorch {metrics['torch_shape']}"
                
            if "nan_mismatch_count" in metrics:
                error_msg += f"\nNaN pattern mismatch: {metrics['nan_mismatch_count']} mismatched positions"
                error_msg += f"\nNumPy NaN count: {metrics['np_nan_count']}, PyTorch NaN count: {metrics['torch_nan_count']}"
                
            if "max_abs_diff" in metrics:
                error_msg += f"\nMax absolute difference: {metrics['max_abs_diff']:.8e} (tolerance: {atol:.8e})"
                
            if "max_rel_diff" in metrics:
                error_msg += f"\nMax relative difference: {metrics['max_rel_diff']:.8e} (tolerance: {rtol:.8e})"
                
            # Include sample values for debugging
            if isinstance(torch_tensor, torch.Tensor) and torch_tensor.numel() > 0:
                torch_array = torch_tensor.detach().cpu().numpy()
                
                # Find index of maximum difference
                if np_array.size > 0 and torch_array.size > 0:
                    abs_diff = np.abs(np_array - torch_array)
                    if abs_diff.size > 0:
                        max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
                        error_msg += f"\nMax difference at index {max_diff_idx}:"
                        error_msg += f"\n  NumPy: {np_array[max_diff_idx]}"
                        error_msg += f"\n  PyTorch: {torch_array[max_diff_idx]}"
            
            # Raise AssertionError with detailed message
            raise AssertionError(error_msg)


class ModelState:
    """
    Utilities for capturing and injecting model state.
    
    This class provides methods for capturing state from a model instance
    and injecting it into another instance.
    """
    
    # Default critical attributes to capture
    DEFAULT_ATTRIBUTES = [
        'hsampling', 'ksampling', 'lsampling', 'n_asu', 'n_atoms_per_asu',
        'n_dof_per_asu', 'n_dof_per_asu_actual', 'n_cell', 'id_cell_ref',
        'map_shape', 'A_inv', 'kvec', 'kvec_norm', 'Amat', 'Linv'
    ]
    
    @staticmethod
    def capture_model_state(model: Any, attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Capture relevant state from a model instance.
        
        Args:
            model: Model instance to capture state from
            attributes: List of attribute names to capture, or None for defaults
            
        Returns:
            Dictionary with captured state
        """
        # If attributes is None, use default critical attributes list
        if attributes is None:
            attributes = ModelState.DEFAULT_ATTRIBUTES
            
        # Create state dictionary with attribute name -> value mapping
        state = {}
        
        for attr in attributes:
            if hasattr(model, attr):
                value = getattr(model, attr)
                state[attr] = value
                
        # Capture model parameters if model has them
        if hasattr(model, 'parameters'):
            try:
                params = list(model.parameters())
                if params:
                    state['_parameters'] = params
            except (TypeError, AttributeError):
                # Not a callable or doesn't return an iterable
                pass
                
        return state
    
    @staticmethod
    def inject_model_state(model: Any, 
                          state: Dict[str, Any], 
                          to_tensor: bool = True, 
                          device: Optional[torch.device] = None) -> None:
        """
        Inject captured state into a model instance.
        
        Args:
            model: Model instance to inject state into
            state: State dictionary from capture_model_state
            to_tensor: Whether to convert NumPy arrays to tensors
            device: Device to place tensors on, or None for model's device
        """
        # Import copy for deep copying mutable objects
        import copy
        
        # Determine target device (from model.device if device=None)
        if device is None and hasattr(model, 'device'):
            device = model.device
            
        # For each attribute in state, set corresponding attribute in model
        for attr, value in state.items():
            # Skip special attributes
            if attr.startswith('_'):
                continue
                
            # Convert NumPy arrays to tensors if to_tensor=True
            if to_tensor and isinstance(value, np.ndarray):
                # Convert to tensor and place on device
                tensor_value = torch.tensor(value, device=device, dtype=torch.float32)
                
                # Set requires_grad=True for floating point tensors
                if tensor_value.is_floating_point():
                    tensor_value.requires_grad_(True)
                    
                setattr(model, attr, tensor_value)
            elif isinstance(value, torch.Tensor) and device is not None:
                # Move existing tensor to the specified device
                setattr(model, attr, value.to(device))
            elif isinstance(value, (list, dict, set)):
                # Create a deep copy for mutable objects to avoid reference issues
                setattr(model, attr, copy.deepcopy(value))
            else:
                # Set attribute directly
                setattr(model, attr, value)
