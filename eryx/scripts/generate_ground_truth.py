#!/usr/bin/env python3
"""
Ground truth data generation script for PyTorch port.

This script runs the NumPy implementation with different parameter sets
to generate ground truth data for testing the PyTorch implementation.
"""

import os
import sys
import logging
import time
import numpy as np

# Import the configuration to ensure debug mode is enabled
from eryx.autotest_config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("ground_truth_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import after setting DEBUG_MODE
# Add the project root to the path so we can import run_debug.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from run_debug import run_np
from eryx.autotest_config import config

def main():
    """
    Main function to generate ground truth data.
    """
    start_time = time.time()
    logging.info("Starting ground truth data generation")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Ensure log directory exists
    log_dir = "logs"  # Use the default log directory
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Log files will be saved to: {log_dir}")
    
    # Run the NumPy implementation once
    run_start = time.time()
    logging.info("Running NumPy implementation")
    try:
        run_np()
        logging.info(f"Completed run in {time.time() - run_start:.2f} seconds")
    except Exception as e:
        logging.error(f"Error in run: {e}")
    
    # Verify log files were created
    log_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(log_files)} log files")
    
    total_time = time.time() - start_time
    logging.info(f"Ground truth data generation completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
