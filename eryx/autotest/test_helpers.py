import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Type, Tuple

def load_test_state(logger, module_name: str, class_name: str, method_name: str, before: bool = True) -> Dict[str, Any]:
    """
    Load state from log file with flexible path handling.
    
    Args:
        logger: Logger instance
        module_name: Name of the module
        class_name: Name of the class
        method_name: Name of the method
        before: If True, load before state, else after state
        
    Returns:
        Dictionary with state data
        
    Raises:
        FileNotFoundError: If state log file is not found in any format
    """
    import os
    import glob
    import logging
    
    prefix = "_state_before_" if before else "_state_after_"
    
    # Try several possible path formats
    possible_paths = [
        # Format 1: module.class.state_before_method
        f"logs/{module_name}.{class_name}.{prefix}{method_name}.log",
        
        # Format 2: module.method.class.state_before_method
        f"logs/{module_name}.{method_name}.{class_name}.{prefix}{method_name}.log",
        
        # Format 3: module.method.class.state_before_
        f"logs/{module_name}.{method_name}.{class_name}.{prefix}_.log",
        
        # Format 4: module.class.state_before_
        f"logs/{module_name}.{class_name}.{prefix}_.log"
    ]
    
    # Log the paths we're trying
    logging.debug(f"Trying to load state log from the following paths:")
    for i, path in enumerate(possible_paths):
        logging.debug(f"  [{i+1}] {path}")
    
    # Try each path
    errors = []
    for path in possible_paths:
        try:
            # Check if file exists before trying to load
            if os.path.exists(path):
                logging.debug(f"Found log file: {path}")
                return logger.loadStateLog(path)
            else:
                errors.append(f"File not found: {path}")
        except Exception as e:
            errors.append(f"Error loading {path}: {str(e)}")
    
    # If we get here, none of the paths worked
    # Find any matching log files for better error message
    matching_logs = glob.glob(f"logs/*{class_name}*{method_name}*.log")
    
    error_msg = (
        f"Could not load state log for {module_name}.{class_name}.{method_name}\n"
        f"Tried paths: {possible_paths}\n"
        f"Available matching logs: {matching_logs}"
    )
    
    raise FileNotFoundError(error_msg)

def build_test_object(torch_class: Type, state_data: Dict[str, Any], device: Optional[torch.device] = None) -> Any:
    """
    Build a properly initialized test object from state data.
    
    Args:
        torch_class: PyTorch class to instantiate
        state_data: State dictionary from log file
        device: Optional device override
        
    Returns:
        Initialized instance with proper structure
        
    Example:
        model = build_test_object(OnePhonon, before_state, device=torch.device('cpu'))
    """
    from eryx.autotest.state_builder import StateBuilder
    builder = StateBuilder(device=device)
    return builder.build(torch_class, state_data)

def verify_gradient_flow(tensor: torch.Tensor, source_tensor: torch.Tensor, rtol: float = 1e-5) -> bool:
    """
    Verify gradient flow from tensor to source_tensor.
    
    Args:
        tensor: Output tensor
        source_tensor: Input tensor where gradients should flow to
        rtol: Relative tolerance for gradient verification
        
    Returns:
        True if gradients flow correctly, False otherwise
        
    Example:
        loss = torch.sum(model.kvec)
        loss.backward()
        assert verify_gradient_flow(model.kvec, model.model.A_inv)
    """
    # Check tensor requires grad
    if not tensor.requires_grad:
        print("Output tensor doesn't require gradients")
        return False
    
    # Check source tensor requires grad
    if not source_tensor.requires_grad:
        print("Source tensor doesn't require gradients")
        return False
    
    # Create loss and backpropagate
    loss = torch.sum(tensor)
    loss.backward()
    
    # Check if source_tensor has gradients
    if source_tensor.grad is None:
        print("No gradients flowed to source tensor")
        return False
    
    # Check if gradients are non-zero
    grad_sum = torch.sum(torch.abs(source_tensor.grad))
    if grad_sum < rtol:
        print(f"Gradients too small: {grad_sum.item()}")
        return False
    
    return True

def verify_tensor_matches(tensor: torch.Tensor, expected: np.ndarray, 
                         rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Verify tensor values match expected values within tolerance.
    
    Args:
        tensor: PyTorch tensor to verify
        expected: NumPy array with expected values
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if tensor matches expected values, False otherwise
        
    Example:
        assert verify_tensor_matches(model.kvec, expected_kvec)
    """
    # Convert tensor to numpy
    tensor_np = tensor.detach().cpu().numpy()
    
    # Check shapes match
    if tensor_np.shape != expected.shape:
        print(f"Shape mismatch: {tensor_np.shape} vs {expected.shape}")
        return False
    
    # Compare values with tolerance
    return np.allclose(tensor_np, expected, rtol=rtol, atol=atol)

def run_state_based_test(test_obj, torch_class: Type, module_name: str, 
                        class_name: str, method_name: str) -> Tuple[bool, str]:
    """
    Run a standard state-based test.
    
    Args:
        test_obj: Test case instance (with logger, torch_testing attributes)
        torch_class: PyTorch class to test
        module_name: Module name
        class_name: Class name 
        method_name: Method name
        
    Returns:
        Tuple of (success, error_message)
        
    Example:
        success, message = run_state_based_test(self, OnePhonon, 'eryx.models', 
                                              'OnePhonon', '_build_kvec_Brillouin')
        self.assertTrue(success, message)
    """
    try:
        # 1. Load state data
        before_state = load_test_state(test_obj.logger, module_name, class_name, method_name, before=True)
        after_state = load_test_state(test_obj.logger, module_name, class_name, method_name, before=False)
        
        # 2. Build test object
        model = build_test_object(torch_class, before_state, device=test_obj.device)
        
        # 3. Call method
        method = getattr(model, method_name)
        method()
        
        # 4. Verify results
        # Create a dictionary of actual state
        actual_state = {}
        for attr_name in dir(model):
            if attr_name.startswith('_') or callable(getattr(model, attr_name)):
                continue
            attr_value = getattr(model, attr_name)
            if isinstance(attr_value, torch.Tensor):
                actual_state[attr_name] = attr_value.detach().cpu().numpy()
            else:
                actual_state[attr_name] = attr_value
        
        # 5. Compare with expected state
        if not test_obj.torch_testing.compareStates(after_state, actual_state):
            return False, "State comparison failed"
        
        return True, ""
    except Exception as e:
        return False, str(e)
def ensure_tensor(value, device=None):
    """
    Ensure value is a PyTorch tensor with gradients.
    
    Args:
        value: Value to convert (can be tensor or ndarray)
        device: Device to place tensor on
        
    Returns:
        PyTorch tensor with proper gradients and device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    if isinstance(value, torch.Tensor):
        # Already a tensor, ensure device and gradients
        tensor = value.to(device=device)
        if tensor.dtype.is_floating_point and not tensor.requires_grad:
            tensor.requires_grad_(True)
        return tensor
    elif isinstance(value, np.ndarray):
        # Convert ndarray to tensor
        tensor = torch.tensor(value, device=device)
        if tensor.dtype.is_floating_point:
            tensor.requires_grad_(True)
        return tensor
    elif isinstance(value, list) and all(isinstance(x, (int, float, np.number)) for x in value):
        # Convert list of numbers to tensor
        tensor = torch.tensor(value, device=device)
        if tensor.dtype.is_floating_point:
            tensor.requires_grad_(True)
        return tensor
    
    # Fallback to original value
    return value

def inspect_state_attribute(state_dict: Dict[str, Any], attr_path: str, detailed: bool = True) -> Dict[str, Any]:
    """
    Inspect a specific attribute in a state dictionary.
    
    Args:
        state_dict: State dictionary from load_test_state
        attr_path: Path to attribute (e.g., 'model.A_inv')
        detailed: Whether to return detailed info
        
    Returns:
        Dictionary with inspection results
    """
    parts = attr_path.split('.')
    current = state_dict
    
    # Track our path
    results = {
        'found': False,
        'path': attr_path,
        'partial_path': '',
        'type': None,
        'value': None,
        'details': {}
    }
    
    # Navigate through the path
    for i, part in enumerate(parts):
        results['partial_path'] = '.'.join(parts[:i+1])
        
        if not isinstance(current, dict) or part not in current:
            return results
        
        current = current[part]
    
    # We found the attribute
    results['found'] = True
    results['type'] = type(current).__name__
    
    # Add appropriate details based on type
    if isinstance(current, dict):
        results['details']['keys'] = list(current.keys())
        if '__type__' in current:
            results['details']['serialized_type'] = current['__type__']
        results['value'] = f"Dict with {len(current)} keys"
    elif isinstance(current, np.ndarray):
        results['details']['shape'] = current.shape
        results['details']['dtype'] = str(current.dtype)
        results['value'] = f"Array shape {current.shape}"
    elif isinstance(current, str) and len(current) > 100:
        results['value'] = current[:100] + "..."
    else:
        results['value'] = current
        
    return results
