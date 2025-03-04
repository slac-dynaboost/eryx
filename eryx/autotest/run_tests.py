#!/usr/bin/env python3
"""
Test runner for PyTorch implementation.

This script runs tests for the PyTorch implementation of diffuse scattering
calculations, comparing results with the NumPy implementation.
"""

import os
import sys
import argparse
import logging
import importlib
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

def discover_tests(test_dir: str, pattern: str = "test_*.py") -> List[str]:
    """
    Discover test files in the given directory.
    
    Args:
        test_dir: Directory to search for tests
        pattern: Pattern to match test files
        
    Returns:
        List of test module names
    """
    test_modules = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".py") and file.startswith("test_"):
                # Convert path to module name
                rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(test_dir))
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                test_modules.append(module_name)
    return test_modules

def run_test_module(module_name: str) -> bool:
    """
    Run tests in the given module.
    
    Args:
        module_name: Name of the test module
        
    Returns:
        True if all tests pass, False otherwise
    """
    try:
        logger.info(f"Running tests in {module_name}")
        module = importlib.import_module(module_name)
        
        # Find test functions in the module
        test_functions = [
            getattr(module, name) for name in dir(module)
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        
        if not test_functions:
            logger.warning(f"No test functions found in {module_name}")
            return True
        
        # Run each test function
        passed = True
        for test_func in test_functions:
            try:
                logger.info(f"  Running {test_func.__name__}")
                result = test_func()
                if result is False:  # Explicitly check for False
                    logger.error(f"  {test_func.__name__} failed")
                    passed = False
                else:
                    logger.info(f"  {test_func.__name__} passed")
            except Exception as e:
                logger.error(f"  {test_func.__name__} raised exception: {e}")
                passed = False
        
        return passed
    except Exception as e:
        logger.error(f"Error running tests in {module_name}: {e}")
        return False

def run_tests(test_modules: List[str]) -> bool:
    """
    Run all tests in the given modules.
    
    Args:
        test_modules: List of test module names
        
    Returns:
        True if all tests pass, False otherwise
    """
    passed = True
    for module_name in test_modules:
        if not run_test_module(module_name):
            passed = False
    return passed

def main():
    """
    Main entry point for the test runner.
    """
    parser = argparse.ArgumentParser(description="Run tests for PyTorch implementation")
    parser.add_argument("--test-dir", default="tests", help="Directory containing tests")
    parser.add_argument("--pattern", default="test_*_torch.py", help="Pattern to match test files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Discovering tests...")
    test_modules = discover_tests(args.test_dir, args.pattern)
    
    if not test_modules:
        logger.warning(f"No test modules found in {args.test_dir} matching {args.pattern}")
        return 1
    
    logger.info(f"Found {len(test_modules)} test modules")
    for module in test_modules:
        logger.info(f"  {module}")
    
    logger.info("Running tests...")
    if run_tests(test_modules):
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
