import torch
import numpy as np
import os
import logging
import time
from typing import Tuple, Dict, Optional

# --- Import Matplotlib ---
import matplotlib.pyplot as plt
# Set a non-interactive backend if running on a server without display
# import matplotlib
# matplotlib.use('Agg')

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()] # Log to console for simplicity
)

# -- Files and Paths --
PDB_PATH = "tests/pdbs/5zck_p1.pdb"

# -- Simulation Parameters --
SIM_PARAMS = {
    'expand_p1': True,
    'group_by': 'asu',
    'model': 'gnm',
    'gnm_cutoff': 7.0,
    'res_limit': 0.0,
    # Sampling params *MUST* be provided for GNM ADP calc
    'hsampling': [-4, 4, 3],
    'ksampling': [-17, 17, 3],
    'lsampling': [-29, 29, 3],
}
GAMMA_INTRA = 1.5
GAMMA_INTER = 0.7

# -- Back-of-Envelope Parameters --
TARGET_RESOLUTION_A = 2.0
DQ_FRAC = 0.001
REL_ENERGY_JITTER = 0.001
MEAN_PHOTONS_PER_PIXEL = 1e8
NUM_VIS_POINTS = 5

# -- 2D Visualization Parameters --
VISUALIZE_2D = True # Set to True to enable 2D plot generation
SLICE_DIM = 'h'     # Dimension to slice (h, k, or l)
SLICE_VAL = 1.0     # <<-- CHANGE THIS: Choose a non-zero value (e.g., 0.5, 1.0)
H_RANGE_2D = [-4, 4, 3] # Range for h in 2D plot
K_RANGE_2D = [-17, 17, 3] # Range for k in 2D plot
L_RANGE_2D = [-29, 29, 3] # Range for l in 2D plot
SAMPLING_RATE_2D = 5.0 # Sampling rate (points per Miller index) for 2D plot

# -- Computation Settings --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE_REAL = torch.float64
DTYPE_COMPLEX = torch.complex128

# --- Import Eryx Components ---
try:
    from eryx.models_torch import OnePhonon
    # Import PyTorch map utils for grid generation
    from eryx.map_utils_torch import generate_grid
except ImportError as e:
    logging.error(f"Could not import Eryx components. Error: {e}")
    exit()
except Exception as e:
     logging.error(f"An unexpected error occurred during Eryx import: {e}")
     exit()

# --- Helper Function for 1D Visualization ---
def plot_intensity_slice(q_mags_np: np.ndarray, intensities_np: np.ndarray,
                         q0_mag: float, i0: float, q1_mag: float, i1: float,
                         dlnId_dq: float, target_res: float):
    """Generates the 1D visualization plots."""
    # (Keep the existing plot_intensity_slice function as is)
    if not np.all(np.isfinite(q_mags_np)) or not np.all(np.isfinite(intensities_np)):
         logging.warning("Cannot generate 1D plots due to non-finite values in input.")
         return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Intensity Slice around |q| = {q0_mag:.3f} Å⁻¹ (Res = {target_res:.2f} Å)", fontsize=14)

    # --- Plot 1: Linear Intensity vs |q| ---
    ax1 = axes[0]
    ax1.plot(q_mags_np, intensities_np, marker='o', linestyle='-', label='Simulated I(|q|)')
    ax1.plot([q0_mag, q1_mag], [i0, i1], marker='x', markersize=10, color='red', linestyle='--', label='Finite Diff. Points')
    ax1.plot([q0_mag, q1_mag], [i0, i1], color='limegreen', linewidth=2, linestyle='-', label=f'Slope ≈ dI/d|q|')
    ax1.set_xlabel('|q| (Å⁻¹)')
    ax1.set_ylabel('Intensity I(|q|)')
    ax1.set_title('Linear Intensity Slice')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # --- Plot 2: Log Intensity vs |q| ---
    ax2 = axes[1]
    valid_log_mask = intensities_np > 1e-15
    q_mags_log = q_mags_np[valid_log_mask]
    log_intensities = np.full_like(intensities_np, np.nan)
    if np.any(valid_log_mask):
        log_intensities[valid_log_mask] = np.log(intensities_np[valid_log_mask])
    ax2.plot(q_mags_log, log_intensities[valid_log_mask], marker='o', linestyle='-', label='Simulated ln(I(|q|))')

    log_points_q = []
    log_points_i = []
    if i0 > 1e-15: log_points_q.append(q0_mag); log_points_i.append(np.log(i0))
    if i1 > 1e-15: log_points_q.append(q1_mag); log_points_i.append(np.log(i1))

    if len(log_points_q) == 2:
        ax2.plot(log_points_q, log_points_i, marker='x', markersize=10, color='red', linestyle='--', label='Finite Diff. Points')
        ax2.plot(log_points_q, log_points_i, color='limegreen', linewidth=2, linestyle='-', label=f'Slope ≈ dlnI/d|q| = {dlnId_dq:.2e}')
        mid_q = (log_points_q[0] + log_points_q[1]) / 2
        mid_logi = (log_points_i[0] + log_points_i[1]) / 2
        ax2.text(mid_q, mid_logi, f' Slope ≈ {dlnId_dq:.2e} Å', ha='left', va='bottom', color='darkgreen')

    ax2.set_xlabel('|q| (Å⁻¹)')
    ax2.set_ylabel('Log Intensity ln(I(|q|))')
    ax2.set_title('Log Intensity Slice')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("sase_intensity_slice_comparison.png", dpi=150)
    logging.info("Saved 1D intensity slice comparison plot to sase_intensity_slice_comparison.png")
    # plt.show()

# --- Function for 2D Sensitivity Visualization ---
def visualize_2d_sensitivity(pdb_path: str, sim_params: Dict,
                             gamma_intra: float, gamma_inter: float,
                             dq_frac: float, device: torch.device,
                             slice_dim: str, slice_val: float,
                             h_range: list, k_range: list, l_range: list,
                             sampling_rate: float,
                             # --- New Parameters ---
                             rel_energy_jitter: float,
                             noise_level_frac: float):
    """Calculates and visualizes the 2D significance map:
       (Fractional Intensity Change due to Jitter) / (Fractional Noise Level)"""
    logging.info("\n--- Starting 2D Significance Visualization ---") # Changed title
    start_time_2d = time.time()

    # 1. Define the 2D grid sampling parameters (Keep as is)
    h_sampling_2d = (h_range[0], h_range[1], sampling_rate)
    k_sampling_2d = (k_range[0], k_range[1], sampling_rate)
    l_sampling_2d = (l_range[0], l_range[1], sampling_rate)

    # Fix the sliced dimension
    if slice_dim.lower() == 'h':
        h_sampling_2d = (slice_val, slice_val, 1) # Single point, rate doesn't matter
        plane_axes = ('k', 'l')
        logging.info(f"Generating 2D sensitivity map for h={slice_val} plane...")
    elif slice_dim.lower() == 'k':
        k_sampling_2d = (slice_val, slice_val, 1)
        plane_axes = ('h', 'l')
        logging.info(f"Generating 2D sensitivity map for k={slice_val} plane...")
    else: # Default to l=0
        l_sampling_2d = (slice_val, slice_val, 1)
        plane_axes = ('h', 'k')
        logging.info(f"Generating 2D sensitivity map for l={slice_val} plane...")

    # 2. Generate base q-vectors for the slice
    try:
        # Need A_inv to generate grid. Get it from a temporary model instance.
        # Extract non-sampling params from sim_params
        other_sim_params = {k: v for k, v in sim_params.items() if k not in ['hsampling', 'ksampling', 'lsampling']}

        temp_model = OnePhonon(
            pdb_path=pdb_path,
            hsampling=h_sampling_2d,    # Use the specific 2D sampling
            ksampling=k_sampling_2d,    # Use the specific 2D sampling
            lsampling=l_sampling_2d,    # Use the specific 2D sampling
            **other_sim_params,         # Pass the rest of the params
            device=device
        )
        A_inv_tensor = torch.tensor(temp_model.model.A_inv, dtype=DTYPE_REAL, device=device)
        del temp_model # Free memory

        # Use generate_grid from map_utils_torch
        q_grid_slice, map_shape_2d = generate_grid(
            A_inv_tensor, h_sampling_2d, k_sampling_2d, l_sampling_2d, return_hkl=False
        )
        q_grid_slice = q_grid_slice.to(dtype=DTYPE_REAL)
        num_pixels = q_grid_slice.shape[0]
        logging.info(f"Generated base grid for 2D slice: {num_pixels} points, map shape: {map_shape_2d}")

    except Exception as e:
        logging.error(f"Failed to generate base grid for 2D slice: {e}", exc_info=True)
        return

    # 3. Generate perturbed q-vectors
    q_norm_slice = torch.norm(q_grid_slice, dim=1, keepdim=True)
    # Avoid division by zero at the origin q=(0,0,0)
    unit_q_slice = torch.zeros_like(q_grid_slice)
    valid_norm_mask = q_norm_slice.squeeze() > 1e-9
    unit_q_slice[valid_norm_mask] = q_grid_slice[valid_norm_mask] / q_norm_slice[valid_norm_mask]

    dq_mag_slice = q_norm_slice * dq_frac
    q_perturbed_slice = q_grid_slice + dq_mag_slice * unit_q_slice
    q_perturbed_slice = q_perturbed_slice.to(dtype=DTYPE_REAL)

    # 4. Combine base and perturbed vectors for simulation
    q_eval_2d = torch.cat([q_grid_slice, q_perturbed_slice], dim=0)
    logging.info(f"Total points to simulate for 2D map: {q_eval_2d.shape[0]}")

    # 5. Simulate intensities
    try:
        # Extract non-sampling params from sim_params
        other_sim_params = {k: v for k, v in sim_params.items() if k not in ['hsampling', 'ksampling', 'lsampling']}

        gamma_intra_tensor = torch.tensor(gamma_intra, dtype=DTYPE_REAL, device=device)
        gamma_inter_tensor = torch.tensor(gamma_inter, dtype=DTYPE_REAL, device=device)

        eval_params_2d = {
            'pdb_path': pdb_path,
            'q_vectors': q_eval_2d,            # Pass combined points
            'hsampling': sim_params['hsampling'], # <<< Still need the ORIGINAL sampling for GNM ADP
            'ksampling': sim_params['ksampling'], # <<< Still need the ORIGINAL sampling for GNM ADP
            'lsampling': sim_params['lsampling'], # <<< Still need the ORIGINAL sampling for GNM ADP
            'gamma_intra': gamma_intra_tensor,
            'gamma_inter': gamma_inter_tensor,
            'device': device,
            **other_sim_params # Pass other params like expand_p1, group_by, model, gnm_cutoff, res_limit
        }
        model_2d = OnePhonon(**eval_params_2d)
        logging.info("Model initialized for 2D simulation.")

        with torch.no_grad():
            I_eval_2d = model_2d.apply_disorder(use_data_adp=True) # Shape (2 * num_pixels,)
        logging.info("Intensities simulated for 2D map.")

        # Split results
        I_slice = I_eval_2d[:num_pixels]
        I_perturbed_slice = I_eval_2d[num_pixels:]

    except Exception as e:
        logging.error(f"Failed during 2D simulation: {e}", exc_info=True)
        return

    # 6. Calculate derivative map
    # --- Debugging: Check Intensity Stats ---
    I_slice_np = I_slice.cpu().numpy()
    I_perturbed_slice_np = I_perturbed_slice.cpu().numpy()

    logging.info(f"Intensity Stats (I0 - Base): Min={np.nanmin(I_slice_np):.3e}, Max={np.nanmax(I_slice_np):.3e}, Mean={np.nanmean(I_slice_np):.3e}, NaN count={np.sum(np.isnan(I_slice_np))}")
    logging.info(f"Intensity Stats (I1 - Perturbed): Min={np.nanmin(I_perturbed_slice_np):.3e}, Max={np.nanmax(I_perturbed_slice_np):.3e}, Mean={np.nanmean(I_perturbed_slice_np):.3e}, NaN count={np.sum(np.isnan(I_perturbed_slice_np))}")

    dI_np = I_perturbed_slice_np - I_slice_np
    logging.info(f"Difference Stats (dI = I1-I0): Min={np.nanmin(dI_np):.3e}, Max={np.nanmax(dI_np):.3e}, Mean={np.nanmean(dI_np):.3e}, AbsMean={np.nanmean(np.abs(dI_np)):.3e}, Zero count={np.sum(np.isclose(dI_np, 0))}/{dI_np.size}")

    # Check dq magnitude as well
    dq_mag_slice_np = dq_mag_slice.squeeze().cpu().numpy()
    logging.info(f"Step Size Stats (dq): Min={np.nanmin(dq_mag_slice_np):.3e}, Max={np.nanmax(dq_mag_slice_np):.3e}, Mean={np.nanmean(dq_mag_slice_np):.3e}")
    # --- End Debugging ---

    # --- Start NaN Handling in Post-Processing ---
    # 6. Calculate derivative map, handling NaNs explicitly
    I0_np = I_slice.cpu().numpy()
    I1_np = I_perturbed_slice.cpu().numpy()
    dI_np = I1_np - I0_np

    dq_mag_slice_np = dq_mag_slice.squeeze().cpu().numpy()
    dq_mag_slice_safe = np.where(np.abs(dq_mag_slice_np) < 1e-12, 1e-12, dq_mag_slice_np)

    dIdq_slice = dI_np / dq_mag_slice_safe

    # Create a mask for valid I0 values (finite and positive)
    valid_I0_mask = np.isfinite(I0_np) & (I0_np > 1e-15)

    # Initialize the derivative map with NaNs
    dlnIdq_map_flat = np.full_like(I0_np, np.nan)

    # Calculate the derivative ONLY where I0 is valid
    dlnIdq_map_flat[valid_I0_mask] = (1.0 / I0_np[valid_I0_mask]) * dIdq_slice[valid_I0_mask]

    num_nan_derivative = np.sum(np.isnan(dlnIdq_map_flat))
    num_total_pixels = dlnIdq_map_flat.size
    logging.info(f"Calculated derivative map. {num_nan_derivative}/{num_total_pixels} pixels are NaN (due to issues in I0 or calculation).")

    # Log stats for the calculated derivative (ignoring NaNs)
    if num_nan_derivative < num_total_pixels:
         logging.info(f"Derivative Map Stats (dlnI/d|q|, excluding NaNs): Min={np.nanmin(dlnIdq_map_flat):.3e}, Max={np.nanmax(dlnIdq_map_flat):.3e}, Mean={np.nanmean(dlnIdq_map_flat):.3e}")
    else:
         logging.info("Derivative Map Stats (dlnI/d|q|): All NaN")

    # 7. Calculate Fractional Change Map
    # frac_change = abs(d(ln I)/d|q|) * rel_energy_jitter * |q|
    q_norm_slice_np = q_norm_slice.squeeze().cpu().numpy() # Need |q| for frac_change
    frac_change_map_flat = np.full_like(dlnIdq_map_flat, np.nan)
    valid_derivative_mask = ~np.isnan(dlnIdq_map_flat)
    frac_change_map_flat[valid_derivative_mask] = (
        np.abs(dlnIdq_map_flat[valid_derivative_mask]) *
        rel_energy_jitter *
        q_norm_slice_np[valid_derivative_mask] # Use |q| for each pixel
    )

    # 8. Calculate Significance Ratio Map
    significance_map_flat = np.full_like(frac_change_map_flat, np.nan)
    if noise_level_frac > 1e-12: # Avoid division by zero if noise is effectively zero
        valid_frac_change_mask = ~np.isnan(frac_change_map_flat)
        significance_map_flat[valid_frac_change_mask] = (
            frac_change_map_flat[valid_frac_change_mask] / noise_level_frac
        )
    else:
        # If noise is zero, ratio is infinite where frac_change > 0
        positive_frac_change_mask = (~np.isnan(frac_change_map_flat)) & (frac_change_map_flat > 1e-12)
        significance_map_flat[positive_frac_change_mask] = np.inf

    num_nan_significance = np.sum(np.isnan(significance_map_flat))
    logging.info(f"Calculated significance ratio map. {num_nan_significance}/{significance_map_flat.size} pixels are NaN.")
    # --- End Calculation ---

    # 9. Reshape significance map
    significance_map_2d = None # Initialize explicitly
    try:
        # Determine target shape based on slice dimension
        if slice_dim.lower() == 'h':
            reshape_dims = (map_shape_2d[1], map_shape_2d[2])
        elif slice_dim.lower() == 'k':
            reshape_dims = (map_shape_2d[0], map_shape_2d[2])
        else: # 'l' or default
            reshape_dims = (map_shape_2d[0], map_shape_2d[1])

        expected_size = np.prod(reshape_dims)
        actual_size = significance_map_flat.size

        logging.info(f"Attempting reshape: Flat size={actual_size}, Target shape={reshape_dims}, Expected size={expected_size}")

        if actual_size == expected_size:
            # Perform reshape and assign
            temp_reshaped = significance_map_flat.reshape(reshape_dims)
            significance_map_2d = temp_reshaped # Explicit assignment after reshape
            logging.info(f"Successfully reshaped significance map to {significance_map_2d.shape}")
        else:
            # Log mismatch and raise error immediately
            logging.error(f"Reshape size mismatch ({actual_size} vs {expected_size}). Cannot proceed.")
            raise ValueError(f"Reshape size mismatch: flat size {actual_size} != target size {expected_size}")

        # --- Add Check ---
        if significance_map_2d is None:
             logging.error("significance_map_2d is None after reshape attempt, even though no error was raised.")
             raise ValueError("Internal error: significance_map_2d is None.")
        # --- End Check ---

    except Exception as e: # Catch any exception during reshape more broadly
        logging.error(f"Failed during reshape significance map block: {e}", exc_info=True)
        # Optionally: Save the flat map for debugging
        # np.save("debug_significance_map_flat.npy", significance_map_flat)
        # logging.info("Saved flat map to debug_significance_map_flat.npy")
        return # Exit function if reshape fails

    # Ensure significance_map_2d is assigned before proceeding
    if significance_map_2d is None:
         logging.error("Cannot proceed to plotting, significance_map_2d is None.")
         return

    # 10. Plot the 2D map
    plt.figure(figsize=(8, 7))
    plot_data = significance_map_2d # Plot the significance ratio

    # Add logging before copy
    logging.debug(f"Type of plot_data before copy: {type(plot_data)}")
    if plot_data is None:
         logging.error("plot_data is None right before calling .copy() - check assignment logic.")
         return # Prevent error

    cmap = plt.get_cmap('viridis') # Or 'plasma', 'magma'
    # cmap.set_bad(color='grey', alpha=0.5) # Optional: color for NaNs
    plot_label = 'Significance Ratio: (ΔI/I)_jitter / (ΔI/I)_noise'

    # --- MODIFIED VMIN/VMAX CALCULATION ---
    # Create a working copy to avoid modifying the original data
    plot_data_for_limits = plot_data.copy() # Error occurred here

    # Replace infinite values with NaN for percentile calculation
    plot_data_for_limits[np.isinf(plot_data_for_limits)] = np.nan

    # Determine robust color limits, ignoring NaNs
    valid_data = plot_data_for_limits[~np.isnan(plot_data_for_limits)] # Use the version with inf replaced by nan

    if valid_data.size > 0:
        # Use percentiles, cap vmax to emphasize lower values, ensure vmin >= 0
        vmin = max(0, np.percentile(valid_data, 1))  # Ensure vmin is at least 0
        vmax = np.percentile(valid_data, 99) # Cap at 99th percentile

        # Optional: Further cap vmax if values can get very large
        # vmax = min(vmax, 10) # Example: Cap at 10

        if np.isclose(vmin, vmax): vmin = max(0, vmin - 0.1); vmax += 0.1 # Adjust if range is too small
        if vmin >= vmax: vmax = vmin + 0.1 # Ensure vmax > vmin
    else:
        vmin, vmax = 0, 1 # Default range if no valid data
    # --- END MODIFIED VMIN/VMAX CALCULATION ---

    logging.info(f"Plotting Significance Ratio map with vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Plot using the ORIGINAL plot_data (which might still contain inf)
    im = plt.imshow(plot_data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[h_range[0], h_range[1], k_range[0], k_range[1]] if slice_dim.lower() == 'l' else \
                           [k_range[0], k_range[1], l_range[0], l_range[1]] if slice_dim.lower() == 'h' else \
                           [h_range[0], h_range[1], l_range[0], l_range[1]])
    plt.colorbar(im, label=plot_label)
    plt.xlabel(f'{plane_axes[0]} index / Å⁻¹')
    plt.ylabel(f'{plane_axes[1]} index / Å⁻¹')
    plt.title(f'Significance Ratio in {slice_dim}={slice_val} plane (Jitter Effect / Noise Level)')
    # Add contour line where ratio = 1?
    # if valid_data.size > 0:
    #     try:
    #         X, Y = np.meshgrid(np.linspace(h_range[0], h_range[1], reshape_dims[0]),
    #                            np.linspace(k_range[0], k_range[1], reshape_dims[1])) # Adjust axes based on slice
    #         plt.contour(X, Y, plot_data.T, levels=[1.0], colors='red', linestyles='dashed')
    #     except Exception as contour_e:
    #         logging.warning(f"Could not draw contour line: {contour_e}")

    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"sase_sensitivity_map_{slice_dim}{slice_val}.png", dpi=150)
    logging.info(f"Saved 2D sensitivity map plot to sase_sensitivity_map_{slice_dim}{slice_val}.png")
    # plt.show()

    elapsed_time_2d = time.time() - start_time_2d
    logging.info(f"2D Sensitivity Visualization finished in {elapsed_time_2d:.2f} seconds.")


# --- Main Calculation ---
def run_back_of_envelope_refactored():
    """Performs the back-of-the-envelope calculation using standard initialization."""
    start_time = time.time()
    logging.info("="*50)
    logging.info("Back-of-Envelope: Diffuse Intensity Sensitivity to Jitter (Refactored)")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Using PDB: {PDB_PATH}")
    if not os.path.exists(PDB_PATH):
         logging.error(f"PDB file not found at {PDB_PATH}. Exiting.")
         return

    # --- Generate Extended Q-Points for 1D Visualization ---
    q_mag_center = 2.0 * np.pi / TARGET_RESOLUTION_A
    dq_mag_step = q_mag_center * DQ_FRAC # Step size for finite diff
    logging.info(f"Center |q| = {q_mag_center:.3f} Å⁻¹, Step dq = {dq_mag_step:.3e} Å⁻¹")

    center_direction = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE_REAL, device=DEVICE)
    if torch.norm(center_direction) < 1e-9: logging.error("Center direction vector is zero."); return

    q_eval_extended_list = []
    num_half_points = (NUM_VIS_POINTS - 1) // 2
    q_mags_plot = []

    for i in range(-num_half_points, num_half_points + 1):
        # Use the *actual* step size dq_mag_step for generating points
        current_q_mag = q_mag_center + i * dq_mag_step
        q_vec = current_q_mag * center_direction
        q_eval_extended_list.append(q_vec.unsqueeze(0))
        q_mags_plot.append(current_q_mag)

    q_eval_extended = torch.cat(q_eval_extended_list, dim=0).to(dtype=DTYPE_REAL)
    q_mags_plot_np = np.array(q_mags_plot, dtype=np.float64)
    logging.info(f"Generated {NUM_VIS_POINTS} q-vectors for 1D simulation and visualization.")

    idx_q0 = num_half_points
    idx_q1 = num_half_points + 1
    if idx_q1 >= NUM_VIS_POINTS: logging.error("NUM_VIS_POINTS must be at least 3."); return
    q0_mag = q_mags_plot_np[idx_q0]
    q1_mag = q_mags_plot_np[idx_q1]

    # --- Setup and Simulate Model for 1D points ---
    logging.info("Setting up OnePhonon model for 1D evaluation points...")
    try:
        gamma_intra_tensor = torch.tensor(GAMMA_INTRA, dtype=DTYPE_REAL, device=DEVICE)
        gamma_inter_tensor = torch.tensor(GAMMA_INTER, dtype=DTYPE_REAL, device=DEVICE)
        eval_params = {
            'pdb_path': PDB_PATH,
            'q_vectors': q_eval_extended, # Use extended points
            'hsampling': SIM_PARAMS['hsampling'], 'ksampling': SIM_PARAMS['ksampling'], 'lsampling': SIM_PARAMS['lsampling'],
            'expand_p1': SIM_PARAMS['expand_p1'], 'group_by': SIM_PARAMS['group_by'], 'model': SIM_PARAMS['model'],
            'gnm_cutoff': SIM_PARAMS['gnm_cutoff'], 'res_limit': SIM_PARAMS['res_limit'],
            'gamma_intra': gamma_intra_tensor, 'gamma_inter': gamma_inter_tensor, 'device': DEVICE
        }
        model_eval = OnePhonon(**eval_params)
        logging.info("Model setup complete for 1D evaluation points.")
    except Exception as e:
        logging.error(f"Failed to initialize OnePhonon model for 1D evaluation: {e}", exc_info=True)
        return

    # --- Simulate Intensity for 1D points ---
    logging.info(f"Simulating intensities for {NUM_VIS_POINTS} 1D points...")
    try:
        with torch.no_grad():
            I_d_extended = model_eval.apply_disorder(use_data_adp=True)
        I0 = I_d_extended[idx_q0].item()
        I1 = I_d_extended[idx_q1].item()
        if np.isnan(I0) or np.isnan(I1): logging.error(f"Simulation yielded NaN intensity (I0={I0:.4e}, I1={I1:.4e}). Cannot proceed."); return
        if I0 <= 1e-12: logging.warning(f"Intensity at q0 is too low (I0={I0:.4e}). Cannot reliably estimate derivative.")
        logging.info(f"I_d(q0) = {I0:.4e}")
        logging.info(f"I_d(q1) = {I1:.4e}")
        I_d_extended_np = I_d_extended.detach().cpu().numpy()
    except Exception as e:
        logging.error(f"Failed during 1D simulation: {e}", exc_info=True)
        return

    # --- Estimate Logarithmic Derivative (using I0, I1) ---
    dId = I1 - I0
    dq = q1_mag - q0_mag # Actual difference in magnitude
    if abs(dq) < 1e-12: logging.error("Step size dq between q0 and q1 is too small."); return
    if abs(I0) < 1e-15: logging.warning(f"Intensity at q0 is too close to zero ({I0:.4e}). Log derivative calculation skipped."); dlnId_dq = 0.0
    else: dId_dq = dId / dq; dlnId_dq = (1.0 / I0) * dId_dq; logging.info(f"Estimated d(ln I_d)/d|q| ≈ {dlnId_dq:.4e} Å")

    # --- Calculate Fractional Change ---
    frac_change = abs(dlnId_dq) * REL_ENERGY_JITTER * q0_mag
    logging.info(f"Assumed Δω/ω = {REL_ENERGY_JITTER:.4f}")
    logging.info(f"Estimated |d(ln I_d)/d|q|| ≈ {abs(dlnId_dq):.4e} Å")
    logging.info(f"Resulting Fractional Change ΔI_d / I_d ≈ {frac_change:.4f}")

    # --- Interpretation ---
    # (Keep interpretation logic as is)
    logging.info("\n--- Interpretation ---")
    noise_level_frac = 1.0 / np.sqrt(MEAN_PHOTONS_PER_PIXEL) if MEAN_PHOTONS_PER_PIXEL > 0 else float('inf')
    logging.info(f"Assumed fractional Poisson noise level (1/√N) ≈ {noise_level_frac:.4f} (for {MEAN_PHOTONS_PER_PIXEL:.1e} photons)")
    if frac_change < 0.1 * noise_level_frac: logging.info("Conclusion: Jitter effect likely NEGLIGIBLE.")
    elif frac_change < noise_level_frac: logging.info("Conclusion: Jitter effect likely MINOR.")
    elif frac_change < 5 * noise_level_frac: logging.info("Conclusion: Jitter effect MAY BE SIGNIFICANT.")
    else: logging.info("Conclusion: Jitter effect likely SIGNIFICANT.")

    # --- Generate 1D Visualization ---
    logging.info("\n--- Generating 1D Visualization ---")
    try:
        plot_intensity_slice(q_mags_plot_np, I_d_extended_np, q0_mag, I0, q1_mag, I1, dlnId_dq, TARGET_RESOLUTION_A)
    except Exception as e:
        logging.error(f"Failed to generate 1D plots: {e}", exc_info=True)

    # --- Generate 2D Visualization (Optional) ---
    if VISUALIZE_2D:
        try:
            # Pass the calculated noise_level_frac and REL_ENERGY_JITTER
            visualize_2d_sensitivity(
                pdb_path=PDB_PATH,
                sim_params=SIM_PARAMS,
                gamma_intra=GAMMA_INTRA,
                gamma_inter=GAMMA_INTER,
                dq_frac=DQ_FRAC,
                device=DEVICE,
                slice_dim=SLICE_DIM,
                slice_val=SLICE_VAL,
                h_range=H_RANGE_2D,
                k_range=K_RANGE_2D,
                l_range=L_RANGE_2D,
                sampling_rate=SAMPLING_RATE_2D,
                # --- Pass new args ---
                rel_energy_jitter=REL_ENERGY_JITTER,
                noise_level_frac=noise_level_frac
            )
        except Exception as e:
            logging.error(f"Failed during 2D sensitivity visualization: {e}", exc_info=True)

    total_time = time.time() - start_time
    logging.info(f"\nCalculation finished in {total_time:.2f} seconds.")
    logging.info("="*50)


# --- Script Execution ---
if __name__ == "__main__":
    run_back_of_envelope_refactored()
    # Add plt.show() here if you want plots to display interactively after script finishes
    if VISUALIZE_2D:
         try:
             plt.show()
         except Exception as e:
             logging.warning(f"Could not display plots interactively: {e}")
