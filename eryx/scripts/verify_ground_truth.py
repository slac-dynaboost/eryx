#!/usr/bin/env python3
"""
Verification script for ground truth data.

This script checks that all required functions have been captured
in the ground truth data logs.
"""

import os
import sys
import json
import logging
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("ground_truth_verification.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_to_convert_json():
    """Load the to_convert.json file."""
    try:
        with open("to_convert.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading to_convert.json: {e}")
        return {}

def get_function_list(to_convert):
    """Extract a flat list of all functions to convert."""
    functions = []
    
    for module, items in to_convert.items():
        for item_name, item_value in items.items():
            if item_name.startswith("class "):
                class_name = item_name.replace("class ", "")
                for method_name in item_value.keys():
                    functions.append(f"{module}:{class_name}.{method_name}")
            else:
                functions.append(f"{module}:{item_name}")
    
    return functions

def search_log_files(log_dir):
    """Search for log files in the log directory."""
    log_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    return log_files

def extract_function_name(log_file):
    """Extract function name from log file path."""
    # Example: ground_truth_data/eryx.scatter.compute_form_factors.log
    match = re.search(r'([^/]+)\.log$', log_file)
    if match:
        return match.group(1)
    return None

def group_log_files(log_files):
    """Group log files by function name."""
    grouped = defaultdict(list)
    for log_file in log_files:
        func_name = extract_function_name(log_file)
        if func_name:
            grouped[func_name].append(log_file)
    return grouped

def count_function_calls(log_file):
    """Count the number of function calls in a log file."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        # Each function call should have two log entries (args and result)
        return len(lines) // 2
    except Exception as e:
        logging.error(f"Error reading log file {log_file}: {e}")
        return 0

def main():
    """Main verification function."""
    logging.info("Starting ground truth data verification")
    
    # Load the to_convert.json file
    to_convert = load_to_convert_json()
    if not to_convert:
        logging.error("Failed to load to_convert.json")
        return
    
    # Get the list of functions to convert
    expected_functions = get_function_list(to_convert)
    logging.info(f"Found {len(expected_functions)} functions in to_convert.json")
    
    # Search for log files
    log_dir = "ground_truth_data"
    log_files = search_log_files(log_dir)
    logging.info(f"Found {len(log_files)} log files in {log_dir}")
    
    # Group log files by function
    grouped_logs = group_log_files(log_files)
    logging.info(f"Found logs for {len(grouped_logs)} unique functions")
    
    # Count function calls for each function
    function_counts = {}
    for func_name, files in grouped_logs.items():
        total_calls = sum(count_function_calls(file) for file in files)
        function_counts[func_name] = total_calls
    
    # Check for missing functions
    missing_functions = []
    for func in expected_functions:
        module, func_name = func.split(":")
        # Convert from to_convert.json format to log file format
        if "." in func_name:  # It's a class method
            class_name, method_name = func_name.split(".")
            log_func_name = f"{module}.{class_name}.{method_name}"
        else:
            log_func_name = f"{module}.{func_name}"
        
        if log_func_name not in function_counts:
            missing_functions.append(func)
    
    # Print summary
    logging.info("\n=== Ground Truth Data Summary ===")
    logging.info(f"Total functions to convert: {len(expected_functions)}")
    logging.info(f"Functions with log data: {len(function_counts)}")
    logging.info(f"Missing functions: {len(missing_functions)}")
    
    if missing_functions:
        logging.warning("The following functions are missing log data:")
        for func in missing_functions:
            logging.warning(f"  - {func}")
    
    # Print function call counts
    logging.info("\n=== Function Call Counts ===")
    for func_name, count in sorted(function_counts.items(), key=lambda x: x[0]):
        logging.info(f"{func_name}: {count} calls")
    
    logging.info("\nGround truth data verification completed")

if __name__ == "__main__":
    main()
