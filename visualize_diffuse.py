#!/usr/bin/env python3
"""
Visualize the diffuse intensity output from run_debug.py.

This script loads the diffuse intensity file and displays a 2D slice
(using matplotlib). If the output is 3D (e.g. shape (dim_h, dim_k, dim_l)),
you can choose to display a central slice along one dimension.

Usage:
    python visualize_diffuse.py [--dataset {torch,np}]
    
    Optional arguments:
    --dataset {torch,np}  Select which dataset to visualize:
                          - torch: uses "torch_diffuse_intensity.npy" (default)
                          - np: uses "np_diffuse_intensity.npy"
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main(dataset='torch'):
    """
    Visualize diffuse intensity data.
    
    Args:
        dataset (str): Which dataset to use ('torch' or 'np')
    
    Returns:
        numpy.ndarray: The loaded intensity data
    """
    # Determine which dataset file to use
    if dataset.lower() == 'np':
        npy_file = "np_diffuse_intensity.npy"
    else:
        npy_file = "torch_diffuse_intensity.npy"
    
    print(f"Using dataset: {npy_file}")
    
    if not os.path.exists(npy_file):
        print(f"File not found: {npy_file}")
        return None

    # Load the diffuse intensity data
    intensity = np.load(npy_file)
    
    # Check the dimensions of the output
    print(f"Diffuse intensity shape: {intensity.shape}")
    
    # If intensity is 1D, we assume it should be reshaped.
    # For example, if the log indicated map_shape = (25, 103, 175)
    if intensity.ndim == 1:
        # Change the shape here if needed based on your actual grid dimensions.
        map_shape = (25, 103, 175)
        intensity = intensity.reshape(map_shape)
        print(f"Reshaped diffuse intensity to: {intensity.shape}")
    
    # If it's 3D, we can take a central slice for visualization.
    if intensity.ndim == 3:
        # Take the central slice along the first dimension
        slice_index = intensity.shape[0] // 2
        intensity_slice = intensity[slice_index, :, :]
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the intensity slice
        im = ax1.imshow(intensity_slice, cmap='viridis', origin='lower')
        ax1.set_title(f"Diffuse Intensity (Slice {slice_index} of {intensity.shape[0]})")
        ax1.set_xlabel("Axis 2")
        ax1.set_ylabel("Axis 3")
        plt.colorbar(im, ax=ax1, label="Intensity")
        
        # Plot the histogram with log scale on y-axis
        # Filter out extreme values for better visualization
        filtered_data = intensity[intensity < 1e11].ravel()
        ax2.hist(filtered_data, bins=100)
        ax2.set_title("Intensity Histogram")
        ax2.set_xlabel("Intensity Value")
        ax2.set_ylabel("Frequency")
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        # If not 3D, try to plot it directly
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the intensity
        im = ax1.imshow(intensity, cmap='viridis')
        ax1.set_title("Diffuse Intensity")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im, ax=ax1, label="Intensity")
        
        # Plot the histogram with log scale on y-axis
        # Filter out extreme values for better visualization
        filtered_data = intensity[intensity < 1e11].ravel()
        ax2.hist(filtered_data, bins=100)
        ax2.set_title("Intensity Histogram")
        ax2.set_xlabel("Intensity Value")
        ax2.set_ylabel("Frequency")
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    return intensity

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Visualize diffuse intensity data from run_debug.py'
    )
    parser.add_argument(
        '--dataset', 
        choices=['torch', 'np'],
        default='torch',
        help='Select which dataset to visualize (default: torch)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main with the selected dataset
    main(dataset=args.dataset)

