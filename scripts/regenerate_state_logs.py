#!/usr/bin/env python
"""
Script to regenerate state logs with proper serialization.

This script runs the original NumPy implementation with debug mode enabled
to generate state logs with complete array data.
"""

import os
import sys
import numpy as np
import argparse
from typing import List, Optional

# Add parent directory to path to import eryx modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set debug mode environment variable
os.environ["DEBUG_MODE"] = "1"

def setup_logging_dir():
    """Create logs directory if it doesn't exist."""
    os.makedirs("logs", exist_ok=True)
    print(f"Logs will be saved to: {os.path.abspath('logs')}")

def regenerate_kvector_logs():
    """Regenerate state logs for k-vector calculation."""
    print("Regenerating k-vector state logs...")
    
    # Import here to ensure DEBUG_MODE is set first
    from eryx.models import OnePhonon
    
    # Create a model with the correct parameters
    model = OnePhonon(
        pdb_path='tests/pdbs/5zck_p1.pdb',
        hsampling=[-2, 2, 2],
        ksampling=[-2, 2, 2],
        lsampling=[-2, 2, 2],
        expand_p1=True,
        res_limit=0.0
    )
    
    # Print A_inv for verification
    print(f"A_inv shape: {model.model.A_inv.shape}")
    print(f"A_inv:\n{model.model.A_inv}")
    
    # Call the method to generate logs
    model._build_kvec_Brillouin()
    
    # Verify the generated kvec values
    print(f"Generated kvec[0,1,0]: {model.kvec[0,1,0]}")
    print(f"Generated kvec[1,0,0]: {model.kvec[1,0,0]}")
    
    print("k-vector state logs regenerated successfully.")

def regenerate_matrix_logs():
    """Regenerate state logs for matrix construction methods."""
    print("Regenerating matrix construction state logs...")
    
    # Import here to ensure DEBUG_MODE is set first
    from eryx.models import OnePhonon
    
    # Create a model with the correct parameters
    model = OnePhonon(
        pdb_path='tests/pdbs/5zck_p1.pdb',
        hsampling=[-2, 2, 2],
        ksampling=[-2, 2, 2],
        lsampling=[-2, 2, 2],
        expand_p1=True,
        res_limit=0.0
    )
    
    # Call methods to generate logs
    model._build_A()
    model._build_M()
    model._build_M_allatoms()
    
    print("Matrix construction state logs regenerated successfully.")

def regenerate_phonon_logs():
    """Regenerate state logs for phonon calculation methods."""
    print("Regenerating phonon calculation state logs...")
    
    # Import here to ensure DEBUG_MODE is set first
    from eryx.models import OnePhonon
    
    # Create a model with the correct parameters
    model = OnePhonon(
        pdb_path='tests/pdbs/5zck_p1.pdb',
        hsampling=[-2, 2, 2],
        ksampling=[-2, 2, 2],
        lsampling=[-2, 2, 2],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    
    # Call methods to generate logs
    model.compute_gnm_phonons()
    model.compute_hessian()
    
    print("Phonon calculation state logs regenerated successfully.")

def regenerate_covariance_logs():
    """Regenerate state logs for covariance matrix calculation."""
    print("Regenerating covariance matrix state logs...")
    
    # Import here to ensure DEBUG_MODE is set first
    from eryx.models import OnePhonon
    
    # Create a model with the correct parameters
    model = OnePhonon(
        pdb_path='tests/pdbs/5zck_p1.pdb',
        hsampling=[-2, 2, 2],
        ksampling=[-2, 2, 2],
        lsampling=[-2, 2, 2],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    
    # Call method to generate logs
    model.compute_covariance_matrix()
    
    print("Covariance matrix state logs regenerated successfully.")

def regenerate_all_logs():
    """Regenerate all state logs."""
    print("Regenerating all state logs...")
    
    regenerate_kvector_logs()
    regenerate_matrix_logs()
    regenerate_phonon_logs()
    regenerate_covariance_logs()
    
    print("All state logs regenerated successfully.")

def main():
    """Main function to parse arguments and run regeneration."""
    parser = argparse.ArgumentParser(description="Regenerate state logs with proper serialization")
    parser.add_argument("--component", choices=["kvector", "matrix", "phonon", "covariance", "all"],
                      default="all", help="Component to regenerate logs for")
    
    args = parser.parse_args()
    
    # Create logs directory
    setup_logging_dir()
    
    # Run the appropriate regeneration function
    if args.component == "kvector":
        regenerate_kvector_logs()
    elif args.component == "matrix":
        regenerate_matrix_logs()
    elif args.component == "phonon":
        regenerate_phonon_logs()
    elif args.component == "covariance":
        regenerate_covariance_logs()
    else:  # "all"
        regenerate_all_logs()

if __name__ == "__main__":
    main()
