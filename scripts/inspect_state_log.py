#!/usr/bin/env python3
"""
Inspect the contents of a state log file.

This script loads a state log file and prints its contents in a readable format,
showing the structure of the state data and the types of values.

Usage:
    python scripts/inspect_state_log.py PATH [--format {json,yaml,tree}]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eryx.autotest.logger import Logger
from eryx.serialization import ObjectSerializer

def format_value(value: Any, indent: int = 0) -> str:
    """
    Format a value for display.
    
    Args:
        value: Any Python value
        indent: Current indentation level
        
    Returns:
        Formatted string representation
    """
    if value is None:
        return "None"
    elif isinstance(value, (int, float, bool, str)):
        return repr(value)
    elif isinstance(value, (list, tuple)):
        type_name = type(value).__name__
        if len(value) == 0:
            return f"Empty {type_name}"
        elif len(value) > 5:
            return f"{type_name} with {len(value)} items (showing first 5): " + \
                ", ".join(format_value(v) for v in value[:5]) + ", ..."
        else:
            return f"{type_name} with {len(value)} items: " + \
                ", ".join(format_value(v) for v in value)
    elif isinstance(value, dict):
        if len(value) == 0:
            return "Empty dict"
        else:
            type_info = f"Dict with {len(value)} keys"
            if "__type__" in value:
                type_info = f"Object of type {value['__type__']}"
            return type_info
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        # Numpy array or PyTorch tensor
        return f"{type(value).__name__} with shape {value.shape}, dtype {value.dtype}"
    else:
        return f"{type(value).__name__}"

def print_tree(data: Dict[str, Any], indent: int = 0) -> None:
    """
    Print a tree view of the state data.
    
    Args:
        data: State data dictionary
        indent: Current indentation level
    """
    prefix = "  " * indent
    try:
        # Sort items, handling non-string keys
        items = sorted(data.items(), key=lambda x: str(x[0]))
    except Exception:
        # Fallback if sorting fails
        items = data.items()
        
    for key, value in items:
        # Skip internal keys, but only for string keys
        if isinstance(key, str) and key.startswith("__") and key != "__type__":
            continue
            
        # Print key with type info
        print(f"{prefix}{key}: {format_value(value)}")
        
        # Recursively print nested dictionaries
        if isinstance(value, dict) and len(value) > 0:
            print_tree(value, indent + 1)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Inspect state log contents")
    parser.add_argument("path", help="Path to state log file")
    parser.add_argument("--format", choices=["json", "yaml", "tree"], default="tree",
                       help="Output format (default: tree)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.path):
        print(f"Error: File not found: {args.path}")
        return 1
    
    # Load state log
    logger = Logger()
    try:
        state = logger.loadStateLog(args.path)
    except Exception as e:
        print(f"Error loading state log: {e}")
        return 1
    
    # Check format version
    format_version = state.get("__format_version__", 1)
    print(f"State Log Format: v{format_version}")
    print(f"Path: {args.path}")
    print(f"Size: {os.path.getsize(args.path)} bytes")
    print(f"Keys: {len(state)}")
    print()
    
    # Print in requested format
    if args.format == "json":
        # Print indented JSON
        print(json.dumps(state, default=lambda x: str(x), indent=2))
    elif args.format == "yaml":
        # Try to use PyYAML if available
        try:
            import yaml
            print(yaml.dump(state, default_flow_style=False))
        except ImportError:
            print("PyYAML not installed. Install with: pip install pyyaml")
            return 1
    else:  # tree format
        print_tree(state)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
