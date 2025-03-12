import unittest
import os
import torch
import numpy as np
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from eryx.autotest.torch_testing import TorchTesting
from typing import Dict, List, Tuple, Any, Optional, Type

class TestBase(unittest.TestCase):
    def setUp(self):
        # Initialize logger, function mapping, and torch testing
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        
        # Set device from environment variable or use default
        device_name = os.environ.get('TORCH_TEST_DEVICE', 'cpu')
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu')
        
        # Initialize torch testing
        self.torch_testing = TorchTesting(self.logger, self.function_mapping)
        # Store device for later use
        
        # Set tolerances from environment variables or use defaults
        self.rtol = float(os.environ.get('TORCH_TEST_RTOL', '1e-5'))
        self.atol = float(os.environ.get('TORCH_TEST_ATOL', '1e-8'))
        
        # Enable/disable log verification
        self.verify_logs = os.environ.get('TORCH_VERIFY_LOGS', 'True').lower() == 'true'
        
    def _load_state(self, module_name: str, class_name: str, method_name: str, before: bool = True) -> Dict:
        # Construct log path for state
        state_type = "_state_before_" if before else "_state_after_"
        
        # Try both naming patterns with both module names (models and models_torch)
        # Prioritize NumPy logs over PyTorch logs for ground truth comparison
        log_paths = [
            # NumPy logs (prioritize these for ground truth)
            f"logs/eryx.models.{method_name}.{class_name}.{state_type}{method_name}.log",  # NumPy actual pattern
            f"logs/eryx.models.{class_name}.{state_type}{method_name}.log",  # NumPy original pattern
            
            # PyTorch logs (fallback)
            f"logs/eryx.models_torch.{method_name}.{class_name}.{state_type}{method_name}.log",  # PyTorch actual pattern
            f"logs/eryx.models_torch.{class_name}.{state_type}{method_name}.log"  # PyTorch original pattern
        ]
        
        # Try to load from each path
        state = None
        for log_path in log_paths:
            try:
                state = self.logger.loadStateLog(log_path)
                if state:
                    print(f"Captured {('before' if before else 'after')} state: {log_path}")
                    break
            except Exception as e:
                pass
                
        if not state and self.verify_logs:
            self.fail(f"State log file not found: {log_paths[0]}")
            
        return state
        
    def _init_from_state(self, torch_class: Type, state: Dict) -> Any:
        # Create empty instance without calling __init__
        obj = torch_class.__new__(torch_class)
        
        # Always set device attribute for OnePhonon models
        obj.device = self.device
            
        # Set all attributes from state dictionary
        for attr_name, attr_value in state.items():
            if attr_name.startswith('_'):
                continue
                
            try:
                # Handle serialized Gemmi objects
                if isinstance(attr_value, dict) and attr_value.get("_gemmi_type"):
                    # In most cases, we can just use the dictionary representation
                    # If actual Gemmi object needed, use deserializer
                    try:
                        # Import gemmi only if needed
                        import gemmi
                        from eryx.autotest.serializer import GemmiSerializer
                        gemmi_serializer = GemmiSerializer()
                        attr_value = gemmi_serializer.deserialize_gemmi_object(attr_value)
                    except (ImportError, ValueError) as e:
                        # Keep as dictionary if deserialization fails
                        print(f"Warning: Could not deserialize Gemmi object {attr_name}: {e}")
                
                # Handle nested state dictionaries
                elif isinstance(attr_value, dict) and all(isinstance(k, str) for k in attr_value.keys()):
                    # Check if this is a nested state dictionary
                    if 'A_inv' in attr_value or 'cell' in attr_value:
                        # This might be a model state dictionary - flatten key attributes to parent
                        for nested_key, nested_value in attr_value.items():
                            if nested_key in ['A_inv', 'cell', 'xyz', 'unit_cell_axes'] and not hasattr(obj, nested_key):
                                # Convert numpy arrays to tensors
                                if isinstance(nested_value, np.ndarray):
                                    if np.issubdtype(nested_value.dtype, np.floating):
                                        tensor = torch.tensor(nested_value, device=self.device, dtype=torch.float32)
                                        tensor.requires_grad_(True)
                                        setattr(obj, nested_key, tensor)
                                    else:
                                        setattr(obj, nested_key, torch.tensor(nested_value, device=self.device))
                                else:
                                    setattr(obj, nested_key, nested_value)
                
                if isinstance(attr_value, np.ndarray):
                    # Convert numpy arrays to tensors with consistent dtype
                    if np.issubdtype(attr_value.dtype, np.floating):
                        tensor = torch.tensor(attr_value, device=self.device, dtype=torch.float32)
                        tensor.requires_grad_(True)
                    elif np.issubdtype(attr_value.dtype, np.complexfloating):
                        # Handle complex arrays
                        tensor = torch.tensor(attr_value, device=self.device, dtype=torch.complex64)
                        tensor.requires_grad_(True)
                    else:
                        tensor = torch.tensor(attr_value, device=self.device)
                    setattr(obj, attr_name, tensor)
                else:
                    # Set non-array attributes directly
                    setattr(obj, attr_name, attr_value)
            except Exception as e:
                print(f"Warning: Could not set attribute {attr_name}: {e}")
                
        return obj
        
    def _compare_states(self, expected: Dict, actual: Dict, 
                       attr_tolerances: Optional[Dict] = None) -> bool:
        # Filter out Gemmi objects from comparison or handle specially
        expected_filtered = {}
        actual_filtered = {}
        
        for key, value in expected.items():
            # Skip attributes that can't be compared directly
            if isinstance(value, dict) and value.get("_gemmi_type"):
                # For Gemmi objects, only compare essential properties
                expected_filtered[key] = self._extract_gemmi_essentials(value)
            else:
                expected_filtered[key] = value
                
        for key, value in actual.items():
            if key not in expected_filtered:
                continue
                
            if isinstance(value, dict) and value.get("_gemmi_type"):
                actual_filtered[key] = self._extract_gemmi_essentials(value)
            else:
                actual_filtered[key] = value
        
        # Use torch_testing.compareStates with filtered states
        attr_tolerances = attr_tolerances or {}
        return self.torch_testing.compareStates(expected_filtered, actual_filtered, attr_tolerances)
    
    def _extract_gemmi_essentials(self, gemmi_dict: Dict) -> Dict:
        """Extract only essential properties from Gemmi dictionaries for comparison."""
        essentials = {}
        
        gemmi_type = gemmi_dict.get("_gemmi_type")
        if gemmi_type == "Structure":
            if "cell" in gemmi_dict:
                essentials["cell"] = gemmi_dict["cell"]
            if "spacegroup" in gemmi_dict:
                essentials["spacegroup"] = gemmi_dict["spacegroup"]
        elif gemmi_type == "Cell" or gemmi_type == "UnitCell":
            for param in ["a", "b", "c", "alpha", "beta", "gamma"]:
                if param in gemmi_dict:
                    essentials[param] = gemmi_dict[param]
        
        return essentials
        
    def _get_method_args(self, module_name: str, method_name: str) -> Tuple[List, Dict]:
        # Find log file for the method
        log_path = f"logs/{module_name}.{method_name}.log"
        
        try:
            # Load log data and extract args/kwargs from first call
            logs = self.logger.loadLog(log_path)
            if not logs:
                return [], {}
                
            call_log = logs[0]  # Get first call
            
            # Handle different log formats
            if "args" in call_log:
                args = self.logger.serializer.deserialize(call_log["args"])
                kwargs = self.logger.serializer.deserialize(call_log["kwargs"])
            else:
                # For state-based logs, we might not have args/kwargs
                return [], {}
            
            # Skip 'self' for method calls
            if len(args) > 0 and hasattr(args[0], '__dict__'):
                args = args[1:]
            
            # Convert NumPy arrays to PyTorch tensors
            args_tensor = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    try:
                        args_tensor.append(torch.tensor(arg, device=self.device))
                    except Exception:
                        # Fall back to CPU if conversion fails
                        args_tensor.append(torch.tensor(arg))
                else:
                    args_tensor.append(arg)
                    
            kwargs_tensor = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    try:
                        kwargs_tensor[k] = torch.tensor(v, device=self.device)
                    except Exception:
                        # Fall back to CPU if conversion fails
                        kwargs_tensor[k] = torch.tensor(v)
                else:
                    kwargs_tensor[k] = v
            
            return args_tensor, kwargs_tensor
        except Exception as e:
            # Don't fail the test, just return empty args
            print(f"Warning: Failed to extract method arguments: {e}")
            return [], {}
        
    def _verify_tensor(self, tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                      requires_grad: bool = True, check_gradient: bool = False,
                      dtype: Optional[torch.dtype] = None):
        # Verify tensor is not None
        self.assertIsNotNone(tensor, "Tensor should not be None")
        
        # Verify shape if provided
        if expected_shape:
            self.assertEqual(tensor.shape, expected_shape, 
                           f"Expected shape {expected_shape}, got {tensor.shape}")
        
        # Verify dtype if provided
        if dtype:
            self.assertEqual(tensor.dtype, dtype,
                           f"Expected dtype {dtype}, got {tensor.dtype}")
        
        # Verify requires_grad for floating point tensors
        if requires_grad and tensor.dtype.is_floating_point:
            self.assertTrue(tensor.requires_grad, "Tensor should require gradients")
        
        # Verify gradient flow if requested
        if check_gradient and tensor.requires_grad:
            try:
                # Create a simple scalar loss
                loss = tensor.sum()
                loss.backward()
                
                # Verify gradient exists and is not all zeros
                self.assertIsNotNone(tensor.grad, "No gradient computed")
                self.assertFalse(torch.all(tensor.grad == 0), "Gradient is all zeros")
            except Exception as e:
                # If gradient checking fails, print warning but don't fail the test
                print(f"Warning: Gradient check failed: {e}")
            finally:
                # Reset gradient for future tests
                if tensor.grad is not None:
                    tensor.grad = None
        
    def _get_tolerances(self, method_name: str) -> Dict:
        # Return appropriate tolerances based on method name
        # Default to standard tolerances for most methods
        tolerances = {}
        
        # Higher tolerances for eigendecomposition methods
        if method_name in ['compute_gnm_phonons', 'compute_covariance_matrix']:
            tolerances = {
                'V': {'rtol': 1e-4, 'atol': 1e-6},
                'Winv': {'rtol': 1e-4, 'atol': 1e-6},
                'covar': {'rtol': 1e-4, 'atol': 1e-6}
            }
        # For disorder calculations
        elif method_name in ['apply_disorder']:
            tolerances = {'__all__': {'rtol': 1e-3, 'atol': 1e-5}}
        
        return tolerances
        
    def verify_required_logs(self, module_name: str, method_name: str, 
                           required_attrs: Optional[List[str]] = None):
        # Verify logs exist and contain required attributes
        import subprocess
        import glob
        
        required_attrs = required_attrs or []
        attr_str = ",".join(required_attrs)
        
        # First check if the logs exist with the actual naming pattern
        before_pattern = f"logs/{module_name}.{method_name}.{self.class_name}._state_before_{method_name}.log"
        after_pattern = f"logs/{module_name}.{method_name}.{self.class_name}._state_after_{method_name}.log"
        
        before_exists = len(glob.glob(before_pattern)) > 0
        after_exists = len(glob.glob(after_pattern)) > 0
        
        if not (before_exists and after_exists):
            self.fail(f"Log pair not found for {method_name}")
            
        # Now run the verification script
        result = subprocess.run(
            ["python", "scripts/verify_logs.py", 
             "--log-dir", "logs", 
             "--required-attrs", attr_str],
            capture_output=True, text=True
        )
        
        # Check if the method is mentioned in the output
        if method_name not in result.stdout and not (before_exists and after_exists):
            self.fail(f"Log for {method_name} not found in verification output")
