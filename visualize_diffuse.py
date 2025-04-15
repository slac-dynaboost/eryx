#!/usr/bin/env python3
"""
Visualize the diffuse intensity output from run_debug.py.

This script loads the diffuse intensity file and displays 2D slices
and a histogram of the intensity values.

Usage:
    python visualize_diffuse.py [--dataset {torch,np,arbq}]

    Optional arguments:
    --dataset {torch,np,arbq}  Select which dataset to visualize:
                               - torch: uses "torch_diffuse_intensity.npy" (PyTorch Grid) (default)
                               - np: uses "np_diffuse_intensity.npy" (NumPy Grid)
                               - arbq: uses "arb_q_diffuse_intensity.npy" (PyTorch Arbitrary-Q)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_map_shape_from_reference(ref_files=["torch_diffuse_intensity.npy", "np_diffuse_intensity.npy"]):
    """Try to load reference files to infer the 3D map shape."""
    for ref_file in ref_files:
        if os.path.exists(ref_file):
            try:
                ref_data = np.load(ref_file)
                if ref_data.ndim == 3:
                    logging.info(f"Inferred map shape {ref_data.shape} from {ref_file}")
                    return ref_data.shape
            except Exception as e:
                logging.warning(f"Could not load or read shape from {ref_file}: {e}")
    return None

def main(dataset='torch'):
    """
    Load, potentially reshape, and visualize diffuse intensity data.

    Args:
        dataset (str): Which dataset to use ('torch', 'np', or 'arbq')

    Returns:
        numpy.ndarray: The loaded (and potentially reshaped) intensity data, or None if loading failed.
    """
    # Determine which dataset file to use and set title
    if dataset.lower() == 'np':
        npy_file = "np_diffuse_intensity.npy"
        plot_title_prefix = "NumPy Grid"
    elif dataset.lower() == 'arbq':
        npy_file = "arb_q_diffuse_intensity.npy"
        plot_title_prefix = "PyTorch Arb-Q"
    else: # Default to 'torch'
        npy_file = "torch_diffuse_intensity.npy"
        plot_title_prefix = "PyTorch Grid"

    logging.info(f"Attempting to load dataset: {npy_file}")

    if not os.path.exists(npy_file):
        logging.error(f"File not found: {npy_file}")
        print(f"\nError: Could not find the required dataset file '{npy_file}'.")
        print("Please ensure you have run the corresponding step in 'run_debug.py'.")
        return None

    # Load the diffuse intensity data
    try:
        intensity = np.load(npy_file)
        logging.info(f"Successfully loaded data. Initial shape: {intensity.shape}")
    except Exception as e:
        logging.error(f"Failed to load {npy_file}: {e}")
        return None

    # --- Reshape 1D Arbitrary-Q Data ---
    map_shape = None
    if intensity.ndim == 1:
        logging.info("Loaded 1D data, attempting to reshape to 3D grid.")
        # Try to infer shape from reference files
        map_shape = get_map_shape_from_reference()

        if map_shape:
            expected_size = np.prod(map_shape)
            if intensity.size == expected_size:
                try:
                    intensity = intensity.reshape(map_shape)
                    logging.info(f"Reshaped data to: {intensity.shape}")
                except ValueError as e:
                    logging.error(f"Failed to reshape 1D data to {map_shape}: {e}. Size mismatch?")
                    map_shape = None # Reset map_shape if reshape failed
            else:
                logging.warning(f"Size of 1D data ({intensity.size}) does not match expected size "
                                f"from inferred map shape {map_shape} ({expected_size}). Cannot reshape.")
                map_shape = None # Reset map_shape as it's invalid for this data
        else:
            logging.warning("Could not infer map shape from reference files. Cannot reshape 1D data.")

    # --- Visualization ---
    # If data is 3D (either loaded or reshaped), plot slices
    if intensity.ndim == 3:
        map_shape = intensity.shape # Update map_shape if it was loaded as 3D
        logging.info(f"Visualizing 3D data with shape: {map_shape}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Changed to 3 slices
        fig.suptitle(f'{plot_title_prefix} Diffuse Intensity Central Slices', fontsize=16)

        # Determine reasonable color limits, ignoring NaNs
        valid_data = intensity[~np.isnan(intensity)]
        if valid_data.size > 0:
            vmin = np.percentile(valid_data, 1)
            vmax = np.percentile(valid_data, 99)
            # Handle cases where vmin and vmax are too close or identical
            if np.isclose(vmin, vmax):
                vmin = valid_data.min()
                vmax = valid_data.max()
            if np.isclose(vmin, vmax): # If still identical (e.g., constant data)
                 vmin = vmin * 0.9 if vmin != 0 else -0.1
                 vmax = vmax * 1.1 if vmax != 0 else 0.1
        else:
            vmin, vmax = 0, 1
            logging.warning("No valid (non-NaN) data points found for color limits.")

        # Central indices
        h_mid, k_mid, l_mid = [s // 2 for s in map_shape]

        # Plot slices
        im1 = axes[0].imshow(intensity[h_mid, :, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'H = {h_mid}')
        axes[0].set_xlabel('L index')
        axes[0].set_ylabel('K index')

        im2 = axes[1].imshow(intensity[:, k_mid, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'K = {k_mid}')
        axes[1].set_xlabel('L index')
        axes[1].set_ylabel('H index')

        im3 = axes[2].imshow(intensity[:, :, l_mid], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
        axes[2].set_title(f'L = {l_mid}')
        axes[2].set_xlabel('K index')
        axes[2].set_ylabel('H index')

        # Add colorbar
        fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    elif intensity.ndim == 2:
        logging.info(f"Visualizing 2D data with shape: {intensity.shape}")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(f'{plot_title_prefix} Diffuse Intensity', fontsize=16)
        im = ax.imshow(intensity, cmap='viridis', origin='lower', interpolation='nearest')
        ax.set_title("2D Intensity Map")
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 0")
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    else:
        logging.warning(f"Data is {intensity.ndim}D. Cannot display slices. Plotting histogram only.")
        fig = None # No slice plot

    # --- Histogram Plot ---
    plt.figure(figsize=(8, 6)) # Create a new figure for the histogram
    valid_data = intensity[~np.isnan(intensity)].ravel()

    if valid_data.size > 0:
         # Filter out extreme values for better visualization if needed
         # p999 = np.percentile(valid_data, 99.9)
         # filtered_data = valid_data[valid_data <= p999]
         filtered_data = valid_data # Keep all valid data for now

         plt.hist(filtered_data, bins=100)
         plt.title(f"{plot_title_prefix} Intensity Histogram")
         plt.xlabel("Intensity Value")
         plt.ylabel("Frequency")
         plt.yscale('log')
         plt.grid(True, alpha=0.5)
         plt.tight_layout()
    else:
         plt.title(f"{plot_title_prefix} Intensity Histogram - No Valid Data")
         plt.xlabel("Intensity Value")
         plt.ylabel("Frequency")
         logging.warning("No valid data points found for histogram.")

    plt.show()

    return intensity

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Visualize diffuse intensity data from run_debug.py'
    )
    parser.add_argument(
        '--dataset',
        choices=['torch', 'np', 'arbq'], # Added 'arbq' option
        default='torch',
        help='Select which dataset to visualize: torch (PyTorch Grid), np (NumPy Grid), arbq (PyTorch Arbitrary-Q) (default: torch)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main with the selected dataset
    main(dataset=args.dataset)
