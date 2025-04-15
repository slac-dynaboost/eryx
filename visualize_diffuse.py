#!/usr/bin/env python3
"""
Visualize the diffuse intensity output from run_debug.py.

Provides functions to load and visualize diffuse intensity data saved
by run_debug.py, handling potential reshaping of 1D data.

Usage (as script):
    python visualize_diffuse.py [--dataset {torch,np,arbq}] [--plot-q]

Usage (as module):
    from visualize_diffuse import load_intensity_data
    q_vectors, intensity_3d, map_shape = load_intensity_data('arbq')
    if intensity_3d is not None:
        # Use the 3D intensity array
        print(intensity_3d.shape)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
from typing import Tuple, Optional

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Function ---
def get_map_shape_from_reference(ref_files=["torch_grid_results.npz", "np_results.npz"]):
    """Try to load reference grid NPZ files to infer the 3D map shape."""
    for ref_file in ref_files:
        if os.path.exists(ref_file):
            try:
                with np.load(ref_file) as data:
                    if 'map_shape' in data:
                        shape = tuple(data['map_shape'])
                        # Basic validation of shape tuple
                        if isinstance(shape, tuple) and len(shape) == 3 and all(isinstance(dim, int) for dim in shape):
                             logging.info(f"Inferred map shape {shape} from reference file: {ref_file}")
                             return shape
                        else:
                             logging.warning(f"Invalid map_shape {shape} found in {ref_file}.")
                    else:
                        # Try loading intensity to infer shape if map_shape key missing
                        if 'intensity' in data and data['intensity'].ndim == 3:
                             shape = data['intensity'].shape
                             logging.info(f"Inferred map shape {shape} from intensity array in {ref_file}")
                             return shape
                        else:
                             logging.warning(f"No 'map_shape' key or 3D 'intensity' array found in {ref_file}.")

            except Exception as e:
                logging.warning(f"Could not load or read shape from {ref_file}: {e}")
    logging.warning("Could not find any valid reference file to infer map shape.")
    return None

# --- Core Loading and Reshaping Function ---
def load_intensity_data(dataset: str = 'torch') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int]]]:
    """
    Loads q-vectors and intensity data for a given dataset, attempting to reshape intensity to 3D.

    Args:
        dataset (str): Which dataset to load ('torch', 'np', or 'arbq').

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int]]]:
            - q_vectors: NumPy array of q-vectors (N, 3), or None if loading failed.
            - intensity: NumPy array of intensity values, reshaped to 3D if possible, otherwise flat (N,), or None if loading failed.
            - map_shape: Tuple (H, K, L) if data is 3D, otherwise None.
    """
    intensity = None
    q_vectors = None
    map_shape = None
    plot_title_prefix = "" # For logging context

    # Determine filenames based on dataset
    if dataset.lower() == 'np':
        npz_file = "np_results.npz"
        npy_file = "np_diffuse_intensity.npy"  # Fallback
        plot_title_prefix = "NumPy Grid"
    elif dataset.lower() == 'arbq':
        npz_file = "torch_arbq_results.npz"
        npy_file = "arb_q_diffuse_intensity.npy"  # Fallback
        plot_title_prefix = "PyTorch Arb-Q"
    else: # Default to 'torch'
        npz_file = "torch_grid_results.npz"
        npy_file = "torch_diffuse_intensity.npy"  # Fallback
        plot_title_prefix = "PyTorch Grid"

    logging.info(f"[{plot_title_prefix}] Attempting to load dataset from {npz_file}")

    # Try loading NPZ first
    if os.path.exists(npz_file):
        try:
            with np.load(npz_file) as data:
                intensity = data.get('intensity')
                q_vectors = data.get('q_vectors')
                map_shape_from_file = tuple(data['map_shape']) if 'map_shape' in data else None

                if intensity is not None:
                    logging.info(f"[{plot_title_prefix}] Loaded intensity ({intensity.shape}) from {npz_file}")
                    if q_vectors is not None:
                        logging.info(f"[{plot_title_prefix}] Loaded q_vectors ({q_vectors.shape}) from {npz_file}")
                    if map_shape_from_file:
                        logging.info(f"[{plot_title_prefix}] Found map_shape {map_shape_from_file} in {npz_file}")
                        map_shape = map_shape_from_file # Use shape from file
                else:
                    logging.warning(f"[{plot_title_prefix}] 'intensity' key not found in {npz_file}")

        except Exception as e:
            logging.error(f"[{plot_title_prefix}] Failed to load or read {npz_file}: {e}")
            intensity = None # Ensure intensity is None if loading fails

    # Fallback to NPY if NPZ failed or didn't contain intensity
    if intensity is None:
        if os.path.exists(npy_file):
            logging.warning(f"[{plot_title_prefix}] NPZ load failed or incomplete. Falling back to NPY: {npy_file}")
            try:
                intensity = np.load(npy_file)
                logging.info(f"[{plot_title_prefix}] Successfully loaded NPY data. Shape: {intensity.shape}")
                # Cannot load q_vectors or map_shape from NPY
                q_vectors = None
                map_shape = None
            except Exception as e:
                logging.error(f"[{plot_title_prefix}] Failed to load {npy_file}: {e}")
                return None, None, None # Return None if fallback fails
        else:
            logging.error(f"[{plot_title_prefix}] Neither NPZ file '{npz_file}' nor NPY file '{npy_file}' could be loaded successfully.")
            return None, None, None # Return None if no files found/loaded

    # --- Attempt Reshaping if Intensity is 1D ---
    intensity_reshaped = intensity # Start with the loaded intensity
    if intensity is not None and intensity.ndim == 1:
        logging.info(f"[{plot_title_prefix}] Intensity is 1D, attempting reshape.")

        # Use map_shape from NPZ if available, otherwise infer
        reshape_target_shape = map_shape if map_shape else get_map_shape_from_reference()

        if reshape_target_shape:
            expected_size = np.prod(reshape_target_shape)
            if intensity.size == expected_size:
                try:
                    intensity_reshaped = intensity.reshape(reshape_target_shape)
                    map_shape = reshape_target_shape # Update map_shape to the successful reshape
                    logging.info(f"[{plot_title_prefix}] Reshaped intensity to: {intensity_reshaped.shape}")
                except ValueError as e:
                    logging.error(f"[{plot_title_prefix}] Failed to reshape 1D intensity ({intensity.size}) to {reshape_target_shape} ({expected_size}): {e}. Returning flat array.")
                    map_shape = None # Reset map_shape as reshape failed
            else:
                logging.warning(f"[{plot_title_prefix}] Size of 1D intensity ({intensity.size}) does not match expected size "
                                f"from map_shape {reshape_target_shape} ({expected_size}). Returning flat array.")
                map_shape = None # Reset map_shape
        else:
            logging.warning(f"[{plot_title_prefix}] Could not determine target 3D shape. Returning flat array.")
            map_shape = None

    elif intensity is not None and intensity.ndim == 3:
         map_shape = intensity.shape # Ensure map_shape reflects loaded 3D data

    # Final check on map_shape if intensity is not 3D
    if intensity_reshaped.ndim != 3:
        map_shape = None

    return q_vectors, intensity_reshaped, map_shape


# --- Main Visualization Function (using the loader) ---
def main(dataset='torch', plot_q=False):
    """
    Load, potentially reshape, and visualize diffuse intensity data.

    Args:
        dataset (str): Which dataset to use ('torch', 'np', or 'arbq')
        plot_q (bool): Whether to plot q-vector distribution.
    """
    q_vectors, intensity, map_shape = load_intensity_data(dataset)

    if intensity is None:
        print("Failed to load intensity data. Exiting.")
        return

    # Determine title prefix
    if dataset.lower() == 'np':
        plot_title_prefix = "NumPy Grid"
    elif dataset.lower() == 'arbq':
        plot_title_prefix = "PyTorch Arb-Q"
    else:
        plot_title_prefix = "PyTorch Grid"

    # --- Visualization ---
    # Plot slices only if data is 3D
    slice_fig = None
    if intensity.ndim == 3 and map_shape is not None:
        logging.info(f"Visualizing 3D data with shape: {map_shape}")
        slice_fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        slice_fig.suptitle(f'{plot_title_prefix} Diffuse Intensity Central Slices', fontsize=16)

        # Determine reasonable color limits, ignoring NaNs
        valid_data = intensity[~np.isnan(intensity)]
        if valid_data.size > 0:
            vmin = np.percentile(valid_data, 1)
            vmax = np.percentile(valid_data, 99)
            if np.isclose(vmin, vmax): vmin = valid_data.min(); vmax = valid_data.max()
            if np.isclose(vmin, vmax): vmin = vmin * 0.9 if vmin != 0 else -0.1; vmax = vmax * 1.1 if vmax != 0 else 0.1
        else: vmin, vmax = 0, 1; logging.warning("No valid data points found for color limits.")

        h_mid, k_mid, l_mid = [s // 2 for s in map_shape]

        # Plot slices (using a simplified call)
        im1 = axes[0].imshow(intensity[h_mid, :, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'H = {h_mid}'); axes[0].set_xlabel('L index'); axes[0].set_ylabel('K index')
        im2 = axes[1].imshow(intensity[:, k_mid, :], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'K = {k_mid}'); axes[1].set_xlabel('L index'); axes[1].set_ylabel('H index')
        im3 = axes[2].imshow(intensity[:, :, l_mid], origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
        axes[2].set_title(f'L = {l_mid}'); axes[2].set_xlabel('K index'); axes[2].set_ylabel('H index')

        slice_fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Intensity')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{dataset}_intensity_slices.png', dpi=300)
        logging.info(f"Saved slice comparison to {dataset}_intensity_slices.png")

    elif intensity.ndim == 2: # Handle 2D data if loaded
        logging.info(f"Visualizing 2D data with shape: {intensity.shape}")
        slice_fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        slice_fig.suptitle(f'{plot_title_prefix} Diffuse Intensity', fontsize=16)
        im = ax.imshow(intensity, cmap='viridis', origin='lower', interpolation='nearest')
        ax.set_title("2D Intensity Map"); ax.set_xlabel("Axis 1"); ax.set_ylabel("Axis 0")
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{dataset}_intensity_2d.png', dpi=300)
        logging.info(f"Saved 2D intensity plot to {dataset}_intensity_2d.png")

    else: # Handle 1D data that couldn't be reshaped
        logging.warning(f"Data is {intensity.ndim}D and could not be reshaped to 3D. Cannot display slices.")

    # --- Histogram Plot (Always attempt) ---
    hist_fig = plt.figure(figsize=(8, 6)) # Create a new figure for the histogram
    valid_data = intensity[~np.isnan(intensity)].ravel() # Ravel works for any dimension

    if valid_data.size > 0:
         filtered_data = valid_data # Keep all valid data
         plt.hist(filtered_data, bins=100)
         plt.title(f"{plot_title_prefix} Intensity Histogram")
         plt.xlabel("Intensity Value"); plt.ylabel("Frequency"); plt.yscale('log')
         plt.grid(True, alpha=0.5); plt.tight_layout()
    else:
         plt.title(f"{plot_title_prefix} Intensity Histogram - No Valid Data")
         plt.xlabel("Intensity Value"); plt.ylabel("Frequency")
         logging.warning("No valid data points found for histogram.")

    plt.savefig(f'{dataset}_intensity_histogram.png', dpi=300)
    logging.info(f"Saved histogram plot to {dataset}_intensity_histogram.png")

    # --- Q-Vector Plot (if requested and available) ---
    if plot_q:
        if q_vectors is not None:
            logging.info(f"Plotting q-vector distribution ({q_vectors.shape[0]} points)...")
            try:
                # Q-vector magnitude histogram
                plt.figure(figsize=(10, 8))
                q_magnitudes = np.linalg.norm(q_vectors, axis=1)
                plt.hist(q_magnitudes, bins=50)
                plt.title(f'Q-Vector Magnitude Distribution ({plot_title_prefix})')
                plt.xlabel('|q| (Å⁻¹)'); plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(f'{dataset}_q_magnitude_histogram.png', dpi=300)
                logging.info(f"Saved q-magnitude histogram to {dataset}_q_magnitude_histogram.png")

                # 3D scatter plot (subsample if needed)
                from mpl_toolkits.mplot3d import Axes3D
                max_points = 5000
                if q_vectors.shape[0] > max_points:
                    indices = np.random.choice(q_vectors.shape[0], max_points, replace=False)
                    q_subset = q_vectors[indices]
                else:
                    q_subset = q_vectors

                fig_3d = plt.figure(figsize=(10, 8))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                scatter = ax_3d.scatter(q_subset[:, 0], q_subset[:, 1], q_subset[:, 2],
                           s=2, alpha=0.5, c=np.linalg.norm(q_subset, axis=1))
                ax_3d.set_xlabel('qx (Å⁻¹)'); ax_3d.set_ylabel('qy (Å⁻¹)'); ax_3d.set_zlabel('qz (Å⁻¹)')
                ax_3d.set_title(f'Q-Vector Distribution ({plot_title_prefix})')
                plt.tight_layout()
                plt.savefig(f'{dataset}_q_vector_scatter.png', dpi=300)
                logging.info(f"Saved q-vector scatter plot to {dataset}_q_vector_scatter.png")

            except ImportError:
                logging.warning("Could not import mpl_toolkits.mplot3d. Skipping 3D scatter plot.")
            except Exception as e:
                logging.error(f"Error plotting q-vectors: {e}")
        else:
            logging.warning(f"Q-vectors not available for dataset '{dataset}' (likely loaded from NPY). Cannot plot q-vectors.")

    # Show plots if run as script
    if __name__ == "__main__":
        plt.show()


# --- Script Execution ---
if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Visualize diffuse intensity data from run_debug.py'
    )
    parser.add_argument(
        '--dataset',
        choices=['torch', 'np', 'arbq'],
        default='torch',
        help='Select which dataset to visualize: torch (PyTorch Grid), np (NumPy Grid), arbq (PyTorch Arbitrary-Q) (default: torch)'
    )
    parser.add_argument(
        '--plot-q',
        action='store_true',
        help='Plot q-vector distribution (only works if data was saved with q-vectors in NPZ format)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main visualization function
    main(dataset=args.dataset, plot_q=args.plot_q)
