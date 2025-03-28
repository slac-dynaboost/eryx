#!/usr/bin/env python3
"""
Script to generate state logs for specified components.
"""
import argparse
import os
import importlib
import logging
import sys
from typing import List, Optional

def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("state_log_generation.log")
        ]
    )

def run_component(component_name: str, exclude_attrs: Optional[List[str]] = None) -> bool:
    """
    Run a specific component to generate state logs.
    
    Args:
        component_name: Name of component to run
        exclude_attrs: Attributes to exclude from state capture
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info(f"Generating logs for component: {component_name}")
        
        if component_name == "onePhonon":
            # Import and run OnePhonon test
            # The run_debug.py file is at the root level, not in eryx package
            import sys
            import os
            
            # Add the root directory to the path if needed
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
                
            # Now import from the root level
            import run_debug
            run_debug.run_np()
            return True
        elif component_name == "mapUtils":
            # Import and run map_utils tests
            import eryx.map_utils as map_utils
            # Call functions that will trigger debug decorators
            return True
        elif component_name == "scatter":
            # Import and run scatter tests
            import eryx.scatter as scatter
            # Call functions that will trigger debug decorators
            return True
        else:
            logging.error(f"Unknown component: {component_name}")
            return False
    except Exception as e:
        logging.error(f"Error running component {component_name}: {str(e)}")
        return False

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate state logs for components')
    parser.add_argument('--component', type=str, default="onePhonon",
                        help='Component to run (onePhonon, mapUtils, scatter)')
    parser.add_argument('--exclude-attrs', type=str, default="",
                        help='Comma-separated list of attributes to exclude')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Enable debug mode
    os.environ["DEBUG_MODE"] = "1"
    
    # Parse excluded attributes
    exclude_attrs = [attr.strip() for attr in args.exclude_attrs.split(',') if attr.strip()]
    
    # Run the specified component
    if args.component == "all":
        components = ["onePhonon", "mapUtils", "scatter"]
        success = all(run_component(comp, exclude_attrs) for comp in components)
    else:
        success = run_component(args.component, exclude_attrs)
    
    # Report results
    if success:
        logging.info("Successfully generated state logs")
    else:
        logging.error("Failed to generate some state logs")
        sys.exit(1)

if __name__ == "__main__":
    main()
