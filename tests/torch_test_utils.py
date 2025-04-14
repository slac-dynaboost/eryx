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
    def compare_tensors(input1: Any, 
                        input2: Any, 
                        rtol: float = 1e-5, 
                        atol: float = 1e-8, 
                        check_nans: bool = True, 
                        equal_nan: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare NumPy array with PyTorch tensor with detailed metrics.
        
        Args:
            input1: First input to compare (NumPy array or PyTorch tensor)
            input2: Second input to compare (NumPy array or PyTorch tensor)
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            check_nans: Whether to verify NaN patterns match
            equal_nan: Whether to consider NaN values as equal
            
        Returns:
            Tuple of (bool success, dict metrics) with detailed comparison results
        """
        # Handle None inputs first
        if input1 is None and input2 is None:
            return True, {"message": "Both inputs are None"}
        
        if input1 is None or input2 is None:
            return False, {"message": f"One input is None: input1={input1 is not None}, input2={input2 is not None}"}
        
        # Convert input1 to NumPy array if it's a Tensor
        if isinstance(input1, torch.Tensor):
            array1 = input1.detach().cpu().numpy()
        elif isinstance(input1, np.ndarray):
            array1 = input1
        else:
            # Attempt to convert other types, raise error if not possible
            try:
                array1 = np.array(input1)
            except Exception as e:
                return False, {"message": f"Input 1 type {type(input1)} cannot be converted to NumPy array: {e}"}

        # Convert input2 to NumPy array if it's a Tensor
        if isinstance(input2, torch.Tensor):
            array2 = input2.detach().cpu().numpy()
        elif isinstance(input2, np.ndarray):
            array2 = input2
        else:
            # Attempt to convert other types, raise error if not possible
            try:
                array2 = np.array(input2)
            except Exception as e:
                return False, {"message": f"Input 2 type {type(input2)} cannot be converted to NumPy array: {e}"}
            
        # Check shapes match
        if array1.shape != array2.shape:
            return False, {
                "message": "Shape mismatch",
                "input1_shape": array1.shape,
                "input2_shape": array2.shape
            }
            
        # Handle empty arrays
        if array1.size == 0:
            return True, {"message": "Empty arrays"}
            
        # Handle NaN values
        nans1 = np.isnan(array1)
        nans2 = np.isnan(array2)
        
        if check_nans and not np.array_equal(nans1, nans2):
            nan_mismatch_count = np.sum(nans1 != nans2)
            return False, {
                "message": "NaN pattern mismatch",
                "nan_mismatch_count": nan_mismatch_count,
                "input1_nan_count": np.sum(nans1),
                "input2_nan_count": np.sum(nans2)
            }
            
        # Mask out NaNs for comparison if equal_nan is True
        if equal_nan:
            mask = ~(nans1 | nans2)
            if not np.any(mask):
                return True, {"message": "All values are NaN"}
            masked1 = array1[mask]
            masked2 = array2[mask]
        else:
            # When not using equal_nan, we still need to handle NaNs properly
            # for the comparison to work correctly
            if np.any(nans1) and check_nans:
                # If we've reached here, NaN patterns match, so we can mask them out
                mask = ~nans1
                if not np.any(mask):
                    return True, {"message": "All values are NaN"}
                masked1 = array1[mask]
                masked2 = array2[mask]
            else:
                masked1 = array1
                masked2 = array2
            
        # Calculate metrics
        abs_diff = np.abs(masked1 - masked2)
        max_abs_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
        mean_abs_diff = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0
            
        # Calculate relative difference for non-zero values
        non_zero_mask = masked1 != 0
        if np.any(non_zero_mask):
            rel_diff = abs_diff[non_zero_mask] / np.abs(masked1[non_zero_mask])
            max_rel_diff = float(np.max(rel_diff))
            mean_rel_diff = float(np.mean(rel_diff))
        else:
            max_rel_diff = 0.0
            mean_rel_diff = 0.0
            
        # Check if arrays are close
        try:
            is_close = np.allclose(masked1, masked2, rtol=rtol, atol=atol)
        except:
            # Handle case where comparison might fail due to type issues
            is_close = np.allclose(masked1.astype(np.float32), 
                                  masked2.astype(np.float32), 
                                  rtol=rtol, atol=atol)
        
        # Prepare detailed metrics
        metrics = {
            "success": is_close,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": mean_rel_diff,
            "input1_min": np.min(masked1) if masked1.size > 0 else None,
            "input1_max": np.max(masked1) if masked1.size > 0 else None,
            "input2_min": np.min(masked2) if masked2.size > 0 else None,
            "input2_max": np.max(masked2) if masked2.size > 0 else None
        }
        
        return is_close, metrics
    
    @staticmethod
    def assert_tensors_equal(input1: Any, 
                            input2: Any, 
                            rtol: float = 1e-5, 
                            atol: float = 1e-8, 
                            check_nans: bool = True, 
                            equal_nan: bool = False, 
                            msg: Optional[str] = None) -> None:
        """
        Assert that two inputs are equivalent with detailed error reporting.
        
        Args:
            input1: First input to compare (NumPy array or PyTorch tensor)
            input2: Second input to compare (NumPy array or PyTorch tensor)
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            check_nans: Whether to verify NaN patterns match
            equal_nan: Whether to consider NaN values as equal
            msg: Optional error message prefix
        """
        # Use compare_tensors to get detailed metrics
        success, metrics = TensorComparison.compare_tensors(
            input1, input2, rtol, atol, check_nans, equal_nan
        )
        
        if not success:
            # Create clear error message if comparison fails
            error_msg = msg or "Tensor comparison failed"
            
            if "message" in metrics:
                error_msg += f": {metrics['message']}"
                
            if "input1_shape" in metrics and "input2_shape" in metrics:
                error_msg += f"\nShape mismatch: Input1 {metrics['input1_shape']} vs Input2 {metrics['input2_shape']}"
                
            if "nan_mismatch_count" in metrics:
                error_msg += f"\nNaN pattern mismatch: {metrics['nan_mismatch_count']} mismatched positions"
                error_msg += f"\nInput1 NaN count: {metrics['input1_nan_count']}, Input2 NaN count: {metrics['input2_nan_count']}"
                
            if "max_abs_diff" in metrics:
                error_msg += f"\nMax absolute difference: {metrics['max_abs_diff']:.8e} (tolerance: {atol:.8e})"
                
            if "max_rel_diff" in metrics:
                error_msg += f"\nMax relative difference: {metrics['max_rel_diff']:.8e} (tolerance: {rtol:.8e})"
                
            # Include sample values for debugging
            # Convert both inputs to NumPy arrays for comparison
            array1 = input1.detach().cpu().numpy() if isinstance(input1, torch.Tensor) else np.array(input1)
            array2 = input2.detach().cpu().numpy() if isinstance(input2, torch.Tensor) else np.array(input2)
            
            # Find index of maximum difference
            if array1.size > 0 and array2.size > 0:
                try:
                    abs_diff = np.abs(array1 - array2)
                    if abs_diff.size > 0:
                        max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
                        error_msg += f"\nMax difference at index {max_diff_idx}:"
                        error_msg += f"\n  Input1: {array1[max_diff_idx]}"
                        error_msg += f"\n  Input2: {array2[max_diff_idx]}"
                except Exception as e:
                    error_msg += f"\nCould not compute max difference index: {e}"
            
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


class GridMappingUtils:
    """Utilities for mapping between different grid representations."""

    @staticmethod
    def get_bz_to_full_grid_map(kvec_bz: torch.Tensor, kvec_full: torch.Tensor,
                                device: torch.device, match_atol: float = 1e-12) -> torch.Tensor:
        """
        Finds corresponding full grid indices for Brillouin Zone (BZ) k-vectors by matching coordinates.

        Args:
            kvec_bz: Tensor of BZ k-vectors, shape [n_bz_points, 3].
            kvec_full: Tensor of full grid k-vectors, shape [n_full_points, 3].
            device: The PyTorch device.
            match_atol: Absolute tolerance for matching k-vector components.

        Returns:
            Tensor of shape [n_bz_points] containing the corresponding index in the full grid
            for each BZ point.

        Raises:
            ValueError: If mapping fails for any BZ point or is ambiguous.
        """
        # Ensure tensors are on the correct device and have high precision
        kvec_bz = kvec_bz.to(device=device, dtype=torch.float64)
        kvec_full = kvec_full.to(device=device, dtype=torch.float64)

        n_bz_points = kvec_bz.shape[0]
        full_indices_for_bz = torch.full((n_bz_points,), -1, dtype=torch.long, device=device)
        found_indices = set() # Keep track of mapped full indices to detect duplicates

        print(f"Mapping {n_bz_points} BZ k-vectors to full grid k-vectors...")
        for idx_bz in range(n_bz_points):
            target_k = kvec_bz[idx_bz]

            # Calculate absolute differences (element-wise)
            # Expand target_k for broadcasting: [3] -> [1, 3]
            diff = torch.abs(kvec_full - target_k.unsqueeze(0)) # Shape [n_full_points, 3]

            # Find indices where *all* components are close enough
            # match_mask will have shape [n_full_points]
            match_mask = torch.all(diff < match_atol, dim=1)
            matching_indices_full = torch.where(match_mask)[0]

            if matching_indices_full.numel() == 1:
                idx_full = matching_indices_full[0].item()
                if idx_full in found_indices:
                    print(f"Warning: Duplicate mapping found for BZ index {idx_bz} -> Full index {idx_full}. "
                          f"k_bz={target_k.cpu().numpy()}, k_full={kvec_full[idx_full].cpu().numpy()}")
                    # Decide how to handle: error or allow? For now, allow but warn.
                full_indices_for_bz[idx_bz] = idx_full
                found_indices.add(idx_full)
            elif matching_indices_full.numel() == 0:
                # If no exact match, try nearest neighbor as a fallback check
                distances_sq = torch.sum((kvec_full - target_k)**2, dim=1)
                idx_full_nn = torch.argmin(distances_sq).item()
                min_dist_sq = distances_sq[idx_full_nn].item()

                # Check if nearest neighbor distance is extremely small
                if min_dist_sq < match_atol**2 * 10: # Allow slightly larger tolerance for NN check
                    print(f"Warning: No exact match for BZ index {idx_bz}. Using nearest neighbor index {idx_full_nn} "
                          f"(dist_sq={min_dist_sq:.2e}). k_bz={target_k.cpu().numpy()}, k_full_nn={kvec_full[idx_full_nn].cpu().numpy()}")
                    if idx_full_nn in found_indices:
                         print(f"Warning: Duplicate mapping found (nearest neighbor) for BZ index {idx_bz} -> Full index {idx_full_nn}")
                    full_indices_for_bz[idx_bz] = idx_full_nn
                    found_indices.add(idx_full_nn)
                else:
                    # Raise error if no close match found
                    raise ValueError(f"Mapping failed: No matching full grid index found for BZ index {idx_bz} "
                                     f"(k_bz={target_k.cpu().numpy()}, min_dist_sq={min_dist_sq:.2e})")
            else:
                # Multiple exact matches found - ambiguous
                raise ValueError(f"Mapping failed: Multiple exact matching full grid indices found for BZ index {idx_bz}: "
                                 f"{matching_indices_full.cpu().numpy()}. k_bz={target_k.cpu().numpy()}")


        # Final verification (optional but recommended)
        if torch.any(full_indices_for_bz == -1):
             missing_bz_indices = torch.where(full_indices_for_bz == -1)[0].cpu().numpy()
             raise ValueError(f"Mapping verification failed: Some BZ indices were not mapped: {missing_bz_indices}")
        # Check if the number of unique mapped indices is as expected (usually == n_bz_points)
        # Note: This might not hold if multiple BZ points map to the same full grid point due to symmetry.
        # if len(found_indices) != n_bz_points:
        #     print(f"Warning: Number of unique mapped full indices ({len(found_indices)}) "
        #           f"does not match number of BZ points ({n_bz_points}).")

        print("Successfully computed mapping between BZ indices and full grid indices.")
        # print(f"Mapping (BZ idx -> Full idx): {full_indices_for_bz.cpu().numpy()}") # Optional print
        return full_indices_for_bz
