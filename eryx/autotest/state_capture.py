"""
State capture functionality for testing object state before and after method execution.
"""
import re
import logging
from typing import Any, Dict, List, Optional, Pattern, Set, Union
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from eryx.serialization import ObjectSerializer

class StateCapture:
    """
    Captures the state of Python objects with special handling for PyTorch tensors.
    
    This class provides functionality to capture the state of objects including
    complex nested structures, with proper handling for PyTorch tensors,
    numpy arrays, and other Python types.
    """
    
    def __init__(
        self, 
        max_depth: int = 10, 
        exclude_attrs: Optional[List[str]] = None,
        include_private: bool = False
    ):
        """
        Initialize state capture with configuration options.
        
        Args:
            max_depth: Maximum recursion depth for nested objects
            exclude_attrs: List of attribute patterns to exclude
            include_private: Whether to include private attributes (starting with '_')
        """
        self.max_depth = max_depth
        self.exclude_attrs = exclude_attrs or []
        self.include_private = include_private
        
        # Use the improved ObjectSerializer
        self.serializer = ObjectSerializer()
        
        # Compile attribute pattern regexes for faster matching
        self.exclude_patterns = [re.compile(pattern) for pattern in self.exclude_attrs]
        
    def capture_state_v2(self, obj: Any, current_depth: int = 0) -> Dict[str, Any]:
        """
        Recursively capture the state of an object, including nested objects.
        
        This ensures that nested attributes (like A_inv inside AtomicModel)
        are properly captured and can be restored later.
        
        Args:
            obj: Object to capture state from
            current_depth: Current recursion depth (for internal use)
            
        Returns:
            Dictionary containing serialized object state with nested structure
        """
        # Check recursion limit
        if current_depth >= self.max_depth:
            return {"__max_depth_reached__": True}
        
        # Handle None
        if obj is None:
            return None
            
        # Create filtered state dictionary
        state = {}
        
        # Collect filtered attributes
        for attr_name in dir(obj):
            if not self._should_capture_attr(attr_name):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                if callable(attr_value):
                    continue
                
                # For basic types, store directly
                if isinstance(attr_value, (int, float, bool, str, np.ndarray)) or attr_value is None:
                    state[attr_name] = attr_value
                # For lists, process each item recursively if it's an object
                elif isinstance(attr_value, list):
                    state[attr_name] = [
                        self.capture_state_v2(item, current_depth + 1)
                        if hasattr(item, "__dict__") and not isinstance(item, (int, float, bool, str))
                        else item
                        for item in attr_value
                    ]
                # For dictionaries, process each value recursively if it's an object
                elif isinstance(attr_value, dict):
                    state[attr_name] = {
                        key: self.capture_state_v2(val, current_depth + 1)
                        if hasattr(val, "__dict__") and not isinstance(val, (int, float, bool, str))
                        else val
                        for key, val in attr_value.items()
                    }
                # For objects with a __dict__, capture state recursively
                elif hasattr(attr_value, "__dict__"):
                    state[attr_name] = self.capture_state_v2(attr_value, current_depth + 1)
                else:
                    # For other types, store directly
                    state[attr_name] = attr_value
            except Exception as e:
                logging.warning(f"Error capturing attribute {attr_name}: {str(e)}")
                state[f"__error_{attr_name}__"] = str(e)
        
        return state
    
    def capture_state(self, obj: Any, current_depth: int = 0) -> Dict[str, Any]:
        """
        Capture the state of an object recursively.
        
        Args:
            obj: Object to capture state from
            current_depth: Current recursion depth (for internal use)
            
        Returns:
            Dictionary containing serialized object state
        """
        # Use the v2 implementation for recursive capture
        state = self.capture_state_v2(obj, current_depth)
        
        # Add format version marker
        state["__format_version__"] = 2
        
        return state
    
    def _should_capture_attr(self, attr_name: str) -> bool:
        """
        Determine if an attribute should be captured based on configuration.
        
        Args:
            attr_name: Name of the attribute to check
            
        Returns:
            True if the attribute should be captured, False otherwise
        """
        # Skip private attributes unless explicitly included
        if not self.include_private and attr_name.startswith('_'):
            return False
        
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.match(attr_name):
                return False
        
        # Skip common special attributes
        if attr_name in ('__dict__', '__class__', '__module__', '__weakref__'):
            return False
            
        return True
