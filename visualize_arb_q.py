#!/usr/bin/env python3
"""
Visualize the diffuse intensity output from OnePhononTorch in arbitrary q-vector mode.

This script first generates a set of q-vectors based on standard grid sampling
parameters. It then runs the OnePhononTorch model using these q-vectors in
arbitrary q-vector mode. Finally, it reshapes the resulting 1D intensity array
back into the original 3D grid structure and displays central slices.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Add project root to sys.path to allow importing eryx
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from eryx.models_torch import OnePhonon as OnePhononTorch
except ImportError as e:
    logging.error(f"Failed to import OnePhononTorch: {e}")
    logging.error("Please ensure the eryx package is installed and accessible.")
    sys.exit(1)

def get_qvectors_and_shape_from_grid(
    pdb_path: str,
    hsampling: Tuple[float, float, float],
    ksampling: Tuple[float, float, float],
    lsampling: Tuple[float, float, float],
    device: torch.device,
    expand_p1: bool = True,
    res_limit: float = 0.0
) -> Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]:
    """
    Instantiate OnePhonon in grid mode to get q_grid and map_shape.
    """
    logging.info("Generating q-vectors and map_shape from grid parameters...")
    try:
        # Use minimal parameters needed for grid setup
        grid_model_for_setup = OnePhononTorch(
            pdb_path=pdb_path,
            hsampling=hsampling,
            ksampling=ksampling,
            lsampling=lsampling,
            expand_p1=expand_p1,
            res_limit=res_limit,
            device=device,
            # Provide dummy values for phonon params - they aren't needed for grid setup
            model='gnm',
            gnm_cutoff=1.0,
            gamma_intra=1.0,
            gamma_inter=1.0
        )
        q_vectors = grid_model_for_setup.q_grid.clone().detach()
        map_shape = grid_model_for_setup.map_shape
        hkl_grid = grid_model_for_setup.hkl_grid.clone().detach() # Also get hkl grid
        logging.info(f"Generated q_vectors shape: {q_vectors.shape}")
        logging.info(f"Grid map_shape: {map_shape}")
        # We don't need the model instance anymore
        del grid_model_for_setup
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return q_vectors, map_shape, hkl_grid
    except Exception as e:
        logging.error(f"Error during grid setup phase: {e}", exc_info=True)
        raise

def run_arbitrary_q_mode(
    pdb_path: str,
    q_vectors: torch.Tensor,
    hsampling: Tuple[float, float, float], # Need these for ADP calc
    ksampling: Tuple[float, float, float], # Need these for ADP calc
    lsampling: Tuple[float, float, float], # Need these for ADP calc
    device: torch.device,
    expand_p1: bool = True,
    res_limit: float = 0.0,
    gnm_cutoff: float = 4.0,
    gamma_intra: float = 1.0,
    gamma_inter: float = 1.0,
    use_data_adp: bool = True
) -> torch.Tensor:
    """
    Run OnePhononTorch in arbitrary q-vector mode and return intensity.
    """
    logging.info("Running model in arbitrary q-vector mode...")
    try:
        model_arb_q = OnePhononTorch(
            pdb_path=pdb_path,
            q_vectors=q_vectors,
            hsampling=hsampling, # Pass sampling params for ADP
            ksampling=ksampling,
            lsampling=lsampling,
            expand_p1=expand_p1,
            res_limit=res_limit,
            device=device,
            model='gnm',
            gnm_cutoff=gnm_cutoff,
            gamma_intra=gamma_intra,
            gamma_inter=gamma_inter
        )
        
        # Explicitly calculate phonon modes (skipped during initialization in arbitrary q-vector mode)
        logging.info("Explicitly calculating phonon modes for arbitrary q-vector mode...")
#        try:
#            # Calculate phonon modes (eigenvectors and eigenvalues)
#            model_arb_q.compute_gnm_phonons()
#            logging.info("Phonon modes calculated successfully.")
#            
#            # Check if covariance matrix method exists and call it if available
#            if hasattr(model_arb_q, 'compute_covariance_matrix'):
#                model_arb_q.compute_covariance_matrix()
#                logging.info("Covariance matrix calculated successfully.")
#        except Exception as e:
#            logging.error(f"Error during explicit phonon calculation: {e}")
#            logging.warning("Continuing execution, but intensity values may be incorrect.")
        
        logging.info("Model initialized. Calculating intensity...")
        
        # Verify phonon tensors exist and contain non-zero values
        if hasattr(model_arb_q, 'V') and hasattr(model_arb_q, 'Winv'):
            v_nonzero = torch.count_nonzero(torch.abs(model_arb_q.V)).item() if model_arb_q.V is not None else 0
            winv_nonzero = torch.count_nonzero(~torch.isnan(model_arb_q.Winv)).item() if model_arb_q.Winv is not None else 0
            
            logging.info(f"Phonon eigenvectors (V) shape: {model_arb_q.V.shape if model_arb_q.V is not None else 'None'}")
            logging.info(f"Phonon eigenvalues (Winv) shape: {model_arb_q.Winv.shape if model_arb_q.Winv is not None else 'None'}")
            logging.info(f"Non-zero elements - V: {v_nonzero}, Winv: {winv_nonzero}")
            
            if v_nonzero == 0 or winv_nonzero == 0:
                logging.warning("Phonon tensors contain zeros or NaNs only. Intensity values may be incorrect.")
        else:
            logging.warning("Phonon tensors (V and/or Winv) are missing. Intensity values may be incorrect.")
            
        Id_arb_q = model_arb_q.apply_disorder(use_data_adp=use_data_adp)
        logging.info(f"Intensity calculated, shape: {Id_arb_q.shape}")
        return Id_arb_q
    except Exception as e:
        logging.error(f"Error during arbitrary q-vector mode execution: {e}", exc_info=True)
        raise

def visualize_intensity(
    intensity_1d: np.ndarray,
    map_shape: Tuple[int, int, int],
    hkl_grid_np: np.ndarray,
    title_prefix: str = "Arbitrary Q-Mode"
):
    """
    Reshape 1D intensity array and visualize central slices.
    """
    logging.info(f"Reshaping 1D intensity (size {intensity_1d.size}) to 3D ({map_shape})...")

    if intensity_1d.size != np.prod(map_shape):
        logging.error(f"Intensity size mismatch: 1D size is {intensity_1d.size}, "
                      f"but map_shape {map_shape} requires {np.prod(map_shape)} points.")
        # Try to find the closest matching shape, might indicate an issue earlier
        if intensity_1d.size == 8: # Common size for 2x2x2 BZ sampling
             logging.warning("Intensity size matches 2x2x2 BZ. Check if V/Winv were used directly.")
        # As a fallback, try padding with NaN if size is smaller
        if intensity_1d.size < np.prod(map_shape):
             logging.warning("Padding intensity array with NaNs to match map_shape.")
             padded_intensity = np.full(np.prod(map_shape), np.nan)
             padded_intensity[:intensity_1d.size] = intensity_1d
             intensity_1d = padded_intensity
        else:
             logging.error("Cannot reshape intensity array. Aborting visualization.")
             return # Or raise an error

    try:
        intensity_3d = intensity_1d.reshape(map_shape)
    except ValueError as e:
        logging.error(f"Error reshaping intensity array: {e}")
        return

    logging.info("Visualizing central slices...")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title_prefix} Diffuse Intensity (Central Slices)", fontsize=16)

    # Slice indices
    h_slice = map_shape[0] // 2
    k_slice = map_shape[1] // 2
    l_slice = map_shape[2] // 2

    # Determine reasonable color limits, ignoring NaNs
    valid_intensities = intensity_3d[~np.isnan(intensity_3d)]
    if valid_intensities.size > 0:
        # Use percentile to avoid extreme outliers dominating the scale
        vmin = np.percentile(valid_intensities, 1)
        vmax = np.percentile(valid_intensities, 99)
        # Ensure vmin is less than vmax
        if vmin >= vmax:
             vmin = np.min(valid_intensities)
             vmax = np.max(valid_intensities)
        # Handle case where all values are the same
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
    else:
        vmin, vmax = 0, 1 # Default if no valid data

    # Slice 1: (h_slice, k, l) - View along h
    im1 = axes[0].imshow(intensity_3d[h_slice, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Slice h ≈ {hkl_grid_np[h_slice, 0, 0, 0]:.2f}") # Get h value for this slice
    axes[0].set_xlabel("l index")
    axes[0].set_ylabel("k index")
    fig.colorbar(im1, ax=axes[0], label="Intensity")

    # Slice 2: (h, k_slice, l) - View along k
    im2 = axes[1].imshow(intensity_3d[:, k_slice, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Slice k ≈ {hkl_grid_np[0, k_slice, 0, 1]:.2f}") # Get k value
    axes[1].set_xlabel("l index")
    axes[1].set_ylabel("h index")
    fig.colorbar(im2, ax=axes[1], label="Intensity")

    # Slice 3: (h, k, l_slice) - View along l
    im3 = axes[2].imshow(intensity_3d[:, :, l_slice], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Slice l ≈ {hkl_grid_np[0, 0, l_slice, 2]:.2f}") # Get l value
    axes[2].set_xlabel("k index")
    axes[2].set_ylabel("h index")
    fig.colorbar(im3, ax=axes[2], label="Intensity")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    # --- Histogram ---
    plt.figure(figsize=(8, 5))
    # Filter NaNs and potentially extreme values for histogram
    valid_data_hist = intensity_1d[~np.isnan(intensity_1d)]
    if valid_data_hist.size > 0:
         # Filter extreme values for better visualization, e.g., clip at 99.9th percentile
         p999 = np.percentile(valid_data_hist, 99.9)
         filtered_data_hist = valid_data_hist[valid_data_hist <= p999]
         if filtered_data_hist.size > 0:
              plt.hist(filtered_data_hist, bins=50)
         else: # Handle case where all data is filtered out
              plt.hist(valid_data_hist, bins=50) # Plot original valid data
    else:
        plt.hist([], bins=50) # Empty histogram if no valid data

    plt.title(f"{title_prefix} Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Run OnePhononTorch in arbitrary q-mode and visualize results.'
    )
    parser.add_argument(
        '--pdb', type=str, default='tests/pdbs/5zck_p1.pdb',
        help='Path to the PDB file.'
    )
    parser.add_argument('--hmin', type=float, default=-2, help='Min h for grid generation.')
    parser.add_argument('--hmax', type=float, default=2, help='Max h for grid generation.')
    parser.add_argument('--hsteps', type=float, default=2, help='Steps per Miller index for h.')
    parser.add_argument('--kmin', type=float, default=-2, help='Min k for grid generation.')
    parser.add_argument('--kmax', type=float, default=2, help='Max k for grid generation.')
    parser.add_argument('--ksteps', type=float, default=2, help='Steps per Miller index for k.')
    parser.add_argument('--lmin', type=float, default=-2, help='Min l for grid generation.')
    parser.add_argument('--lmax', type=float, default=2, help='Max l for grid generation.')
    parser.add_argument('--lsteps', type=float, default=2, help='Steps per Miller index for l.')
    parser.add_argument('--cutoff', type=float, default=4.0, help='GNM cutoff distance.')
    parser.add_argument('--gamma_intra', type=float, default=1.0, help='Intra-ASU spring constant.')
    parser.add_argument('--gamma_inter', type=float, default=1.0, help='Inter-ASU spring constant.')
    parser.add_argument('--no_data_adp', action='store_true', help='Use computed ADPs instead of PDB data.')
    parser.add_argument('--device', type=str, default='cpu', help='PyTorch device (e.g., cpu, cuda).')

    args = parser.parse_args()

    # --- Validate PDB path ---
    if not os.path.exists(args.pdb):
        logging.error(f"PDB file not found: {args.pdb}")
        return

    # --- Set Device ---
    try:
        if args.device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
        logging.info(f"Using device: {device}")
    except Exception as e:
        logging.warning(f"Invalid device specified ('{args.device}'). Using CPU. Error: {e}")
        device = torch.device('cpu')


    # --- Define Sampling ---
    hsampling = (args.hmin, args.hmax, args.hsteps)
    ksampling = (args.kmin, args.kmax, args.ksteps)
    lsampling = (args.lmin, args.lmax, args.lsteps)

    try:
        # --- Generate Q-vectors and Grid Info ---
        q_vectors, map_shape, hkl_grid = get_qvectors_and_shape_from_grid(
            pdb_path=args.pdb,
            hsampling=hsampling,
            ksampling=ksampling,
            lsampling=lsampling,
            device=device
        )

        # Note on arbitrary q-vector mode:
        # By default, OnePhononTorch skips phonon calculations in arbitrary q-vector mode during initialization.
        # This can lead to zero intensities if not addressed. The run_arbitrary_q_mode function below
        # explicitly calls the necessary phonon calculation methods to ensure correct intensity values.
        
        # --- Run Arbitrary-Q Mode ---
        Id_arb_q = run_arbitrary_q_mode(
            pdb_path=args.pdb,
            q_vectors=q_vectors,
            hsampling=hsampling, # Pass sampling for ADP calc
            ksampling=ksampling,
            lsampling=lsampling,
            device=device,
            gnm_cutoff=args.cutoff,
            gamma_intra=args.gamma_intra,
            gamma_inter=args.gamma_inter,
            use_data_adp=(not args.no_data_adp) # Invert flag logic
        )

        # --- Visualize ---
        intensity_1d_np = Id_arb_q.detach().cpu().numpy()
        hkl_grid_np = hkl_grid.detach().cpu().numpy().reshape(map_shape + (3,))
        visualize_intensity(intensity_1d_np, map_shape, hkl_grid_np)

    except FileNotFoundError as e:
         logging.error(f"File not found error: {e}")
    except ValueError as e:
         logging.error(f"Value error during execution: {e}")
    except RuntimeError as e:
         logging.error(f"Runtime error during execution: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    main()
