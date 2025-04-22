import torch
import torch.nn as nn
import numpy as np
import os
import logging
import time
from typing import Tuple, Dict, List, Optional

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("beam_effect_analysis.log", mode='w'), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)

# -- Files and Paths --
# !!! REPLACE WITH YOUR ACTUAL PDB PATH !!!
PDB_PATH = "tests/pdbs/5zck_p1.pdb" # Using test PDB for example run

# -- Simulation Parameters --
# !!! ADJUST THESE FOR YOUR SYSTEM !!!
SIM_PARAMS = {
    'expand_p1': True,
    'group_by': 'asu',      # Rigid body level ('asu' or None)
    'res_limit': 2.0,       # Angstrom, example resolution limit
    'model': 'gnm',         # Phonon model ('gnm' or 'rb')
    'gnm_cutoff': 7.0,      # Angstrom, example GNM cutoff
    # Sampling params needed for ADP calculation even in arbitrary-q mode
    # Adjust based on your PDB unit cell and desired BZ sampling density
    'hsampling': [-4, 4, 3], # (min, max, points_per_index)
    'ksampling': [-4, 4, 3],
    'lsampling': [-4, 4, 3],
    'use_data_adp': False   # Use computed ADPs for a consistent baseline
}
# --- Ground Truth and Initial Guess Gamma ---
# !!! SET PLAUSIBLE VALUES FOR YOUR SYSTEM !!!
GAMMA_TRUE_INTRA = 1.5      # Assumed true value for generating data
GAMMA_TRUE_INTER = 0.7      # Assumed true value for generating data
INITIAL_GAMMA_INTRA = 1.0   # Starting point for gradient calculation
INITIAL_GAMMA_INTER = 0.5   # Starting point for gradient calculation

# -- Experimental Geometry ---
# !!! CRITICAL: REPLACE ALL PLACEHOLDERS WITH ACCURATE VALUES !!!
GEOMETRY = {
    'detector_distance_mm': 200.0,
    'pixel_size_mm': 0.1,
    'beam_center_pix': (512, 512), # (x, y) or (fast, slow) - check convention
    'wavelength_avg_a': 1.0,       # Average wavelength in Angstrom
    # Add other necessary params like detector tilt angles (rad), origin offset etc.
    # Example: Assuming detector normal along Z, fast axis X, slow axis Y
    # Adjust these vectors based on your actual setup/coordinate system
    'detector_normal': np.array([0, 0, 1]),
    'detector_fast_axis': np.array([1, 0, 0]),
    'detector_slow_axis': np.array([0, 1, 0]),
}

# -- Beam Model Parameters ---
# !!! REPLACE WITH REALISTIC ESTIMATES FOR YOUR XFEL SOURCE !!!
BEAM_PARAMS = {
    'lambda_avg': GEOMETRY['wavelength_avg_a'],
    'bandwidth_fwhm_frac': 0.002, # e.g., 0.2% FWHM relative bandwidth
    'jitter_std_dev_frac': 0.001, # e.g., 0.1% relative std dev jitter
    'n_lambda_samples': 11,       # Number of wavelengths to sample (odd recommended)
}

# --- Noise Parameters ---
# !!! ESTIMATE BASED ON EXPERIMENT !!!
# Average photon count expected in the diffuse signal regions of interest
MEAN_PHOTONS_PER_PIXEL = 10

# -- Computation Settings --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE_REAL = torch.float64 # Use high precision for simulation robustness
DTYPE_COMPLEX = torch.complex128

# --- Gradient Comparison Thresholds ---
# Adjust these based on how much difference you consider "significant"
COS_SIM_THRESHOLD = 0.95  # Threshold for considering directions similar
REL_DIFF_THRESHOLD = 0.15 # Threshold for considering magnitudes similar (allow 15% diff)

# --- Import Eryx Components ---
try:
    from eryx.models_torch import OnePhonon
    # Import necessary map utils if needed for geometry
    # from eryx.map_utils_torch import some_geometry_function
except ImportError as e:
    logging.error(f"Could not import Eryx components. Make sure Eryx is installed. Error: {e}")
    exit()
except Exception as e:
     logging.error(f"An unexpected error occurred during Eryx import: {e}")
     exit()

# --- Helper Functions ---

def define_pixel_coordinates(geometry: Dict, num_pix_x: int = 50, num_pix_y: int = 50) -> np.ndarray:
    """
    Defines the pixel coordinates for simulation (placeholder).

    Args:
        geometry: Dictionary containing beam center.
        num_pix_x: Number of pixels in x dimension for the grid.
        num_pix_y: Number of pixels in y dimension for the grid.

    Returns:
        Numpy array of pixel coordinates [N, 2].
    """
    logging.info("Defining pixel coordinates for simulation...")
    # Placeholder: Create a rectangular grid of pixels around beam center
    center_x, center_y = geometry['beam_center_pix']
    half_x, half_y = num_pix_x // 2, num_pix_y // 2

    # Create ranges ensuring we include the center +/- half extent
    x_coords = np.arange(center_x - half_x, center_x + half_x + 1)
    y_coords = np.arange(center_y - half_y, center_y + half_y + 1)

    # Check if ranges are valid
    if len(x_coords) == 0 or len(y_coords) == 0:
        logging.error("Pixel coordinate range is empty. Check beam center and num_pix.")
        return np.empty((0, 2), dtype=int)

    xx, yy = np.meshgrid(x_coords, y_coords)
    pixel_coords = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(int)

    # Optional: Add masking logic here (e.g., remove beamstop, detector corners)
    # Example: Remove pixels within 10 pixels of the beam center
    # radius_sq = (pixel_coords[:, 0] - center_x)**2 + (pixel_coords[:, 1] - center_y)**2
    # mask = radius_sq > 10**2
    # pixel_coords = pixel_coords[mask]

    if pixel_coords.shape[0] == 0:
        logging.warning("No valid pixel coordinates defined after potential masking.")
    else:
        logging.info(f"Defined {pixel_coords.shape[0]} pixel coordinates.")
    return pixel_coords


def sample_beam_wavelengths(params: Dict, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Samples wavelengths and weights from the beam model (Gaussian approx)."""
    logging.info(f"Sampling {n_samples} wavelengths...")
    lambda_avg = params['lambda_avg']
    # Convert fractional FWHM/std dev to absolute values
    bw_fwhm = lambda_avg * params['bandwidth_fwhm_frac']
    jt_std_dev = lambda_avg * params['jitter_std_dev_frac']

    sigma_bw = bw_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) if bw_fwhm > 0 else 0 # Convert FWHM to std dev
    sigma_total = np.sqrt(sigma_bw**2 + jt_std_dev**2)

    if n_samples <= 0:
        logging.error("n_lambda_samples must be positive.")
        return np.array([]), np.array([])
    elif n_samples == 1 or sigma_total < 1e-9 * lambda_avg: # Handle single sample or zero width
        logging.info("Using single average wavelength (sigma_total is negligible).")
        lambdas = np.array([lambda_avg])
        weights = np.array([1.0])
    else:
        # Sample points symmetrically around the mean using linspace
        # A more sophisticated sampling (e.g., Gaussian quadrature) could be used
        max_dev = 3.0 * sigma_total # Sample out to ~3 sigma
        lambdas = np.linspace(lambda_avg - max_dev, lambda_avg + max_dev, n_samples)
        # Calculate weights using Gaussian PDF (unnormalized)
        pdf_vals = np.exp(-0.5 * ((lambdas - lambda_avg) / sigma_total)**2)
        weights = pdf_vals / np.sum(pdf_vals) # Normalize weights

    logging.info(f"Sampled lambdas (min/max/avg): {lambdas.min():.5f} / {lambdas.max():.5f} / {np.average(lambdas, weights=weights):.5f} Å")
    logging.info(f"Effective beam sigma: {sigma_total:.5f} Å")
    return lambdas.astype(np.float64), weights.astype(np.float64)


def pixel_to_q(pixel_coords: np.ndarray, wavelength_a: float, geometry: Dict) -> np.ndarray:
    """
    Converts pixel coordinates to reciprocal space q-vectors.

    !!! CRITICAL PLACEHOLDER - NEEDS ACCURATE IMPLEMENTATION !!!
    Replace this with a function reflecting your actual detector geometry.
    """
    func_name = "pixel_to_q"
    logging.warning(f"[{func_name}] Using PLACEHOLDER geometry. Results depend heavily on accurate q-vector calculation!")

    if pixel_coords.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    dist = geometry['detector_distance_mm']
    pix_size = geometry['pixel_size_mm']
    center_x, center_y = geometry['beam_center_pix']
    det_n = geometry.get('detector_normal', np.array([0, 0, 1])).astype(float)
    det_f = geometry.get('detector_fast_axis', np.array([1, 0, 0])).astype(float)
    det_s = geometry.get('detector_slow_axis', np.array([0, 1, 0])).astype(float)

    # Ensure axes are normalized
    det_n /= np.linalg.norm(det_n)
    det_f /= np.linalg.norm(det_f)
    det_s /= np.linalg.norm(det_s)

    # Calculate detector coordinates in mm relative to beam center on detector plane
    x_pix, y_pix = pixel_coords[:, 0], pixel_coords[:, 1]
    x_mm = (x_pix - center_x) * pix_size
    y_mm = (y_pix - center_y) * pix_size

    # Vector from sample interaction point to pixel on detector plane (in mm)
    # Assumes sample is at origin (0,0,0)
    # vec_sp = dist * det_n + x_mm * det_f + y_mm * det_s # Incorrect broadcasting
    vec_sp = (dist * det_n[:, np.newaxis] +
              x_mm * det_f[:, np.newaxis] +
              y_mm * det_s[:, np.newaxis]).T # Shape (N, 3)

    # Incident beam wavevector (along +Z, magnitude 2pi/lambda)
    k_in_mag = 2.0 * np.pi / wavelength_a
    k_in = np.array([0, 0, k_in_mag]) # Assumes beam along +Z

    # Scattered beam wavevector (same magnitude, direction of vec_sp)
    k_out_mag = k_in_mag
    vec_sp_norm = np.linalg.norm(vec_sp, axis=1, keepdims=True)
    # Avoid division by zero for pixels at the exact center (though unlikely)
    vec_sp_norm[vec_sp_norm == 0] = 1.0
    k_out = (k_out_mag / vec_sp_norm) * vec_sp

    # Scattering vector q = k_out - k_in
    q_vectors = k_out - k_in

    return q_vectors.astype(np.float64)


def simulate_noisy_measurement(true_intensity_tensor: torch.Tensor, target_mean_photons: float) -> torch.Tensor:
    """
    Simulates a noisy measurement by sampling from a Poisson distribution.

    Args:
        true_intensity_tensor: Tensor containing the simulated 'true' intensities.
        target_mean_photons: The desired average number of photons per pixel
                             in the valid regions of the output.

    Returns:
        A tensor of the same shape as the input, representing noisy intensity
        values incorporating Poisson sampling statistics. Returns NaNs where the
        input was NaN.
    """
    func_name = "simulate_noisy_measurement"
    logging.info(f"[{func_name}] Simulating measurement with target mean photons = {target_mean_photons:.2f}...")

    if not isinstance(true_intensity_tensor, torch.Tensor):
        logging.error(f"[{func_name}] Input must be a torch.Tensor.")
        return None # Or raise error
    if true_intensity_tensor.numel() == 0:
        logging.warning(f"[{func_name}] Input intensity tensor is empty. Returning empty tensor.")
        return true_intensity_tensor

    # Work with a detached copy for noise simulation
    intensity_no_grad = true_intensity_tensor.detach()
    noisy_intensity_output = torch.full_like(intensity_no_grad, float('nan')) # Initialize with NaNs

    # --- Identify valid (non-NaN, positive) data points for scaling ---
    valid_mask = ~torch.isnan(intensity_no_grad) & (intensity_no_grad > 1e-12) # Use small threshold > 0

    if not torch.any(valid_mask):
        logging.warning(f"[{func_name}] Input intensity tensor contains only NaNs or non-positive values. Cannot simulate noise.")
        nan_mask_original = torch.isnan(true_intensity_tensor)
        return torch.where(nan_mask_original, torch.tensor(float('nan'), device=DEVICE, dtype=DTYPE_REAL), intensity_no_grad)

    valid_intensities = intensity_no_grad[valid_mask]

    # --- Calculate scale factor ---
    current_mean_intensity = torch.mean(valid_intensities)
    # Ensure target_mean_photons is positive
    if target_mean_photons <= 0:
        logging.warning(f"[{func_name}] Target mean photons ({target_mean_photons:.2f}) must be positive. Setting expected counts to 0.")
        expected_counts = torch.zeros_like(valid_intensities)
        scale_factor = 1.0 # Avoid division by zero later
    elif current_mean_intensity <= 1e-12:
        logging.warning(f"[{func_name}] Mean intensity ({current_mean_intensity.item():.2e}) of valid points is near zero. Setting expected counts to 0.")
        expected_counts = torch.zeros_like(valid_intensities)
        scale_factor = 1.0
    else:
        scale_factor = target_mean_photons / current_mean_intensity
        expected_counts = valid_intensities * scale_factor
        logging.info(f"[{func_name}] Scaling intensity by {scale_factor:.4f} to get expected counts.")
        logging.info(f"[{func_name}] Expected counts range (min/max/mean): {expected_counts.min():.2f} / {expected_counts.max():.2f} / {expected_counts.mean():.2f}")

    # --- Sample from Poisson Distribution ---
    # Clamp expected_counts to be non-negative before sampling
    sampled_counts = torch.poisson(torch.clamp(expected_counts, min=0.0))
    logging.info(f"[{func_name}] Sampled counts range (min/max/mean): {sampled_counts.min():.1f} / {sampled_counts.max():.1f} / {sampled_counts.mean():.1f}")

    # --- Rescale counts back to intensity units ---
    if abs(scale_factor) < 1e-12:
         logging.warning(f"[{func_name}] Scale factor is near zero. Setting noisy intensity directly from counts (may change units).")
         noisy_valid_intensity = sampled_counts.to(DTYPE_REAL)
    else:
         noisy_valid_intensity = sampled_counts / scale_factor

    # --- Place noisy data back into the full tensor ---
    noisy_intensity_output[valid_mask] = noisy_valid_intensity.to(DTYPE_REAL)

    logging.info(f"[{func_name}] Noise simulation complete.")
    return noisy_intensity_output


# --- Main Analysis Function ---

def run_analysis():
    """Performs the beam effect sensitivity analysis."""
    start_time = time.time()
    logging.info("="*60)
    logging.info("Starting XFEL Jitter/Bandwidth Sensitivity Analysis...")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Using PDB: {PDB_PATH}")
    if not os.path.exists(PDB_PATH):
         logging.error(f"PDB file not found at {PDB_PATH}. Exiting.")
         return

    # 1. Define Simulation Geometry & Sample Wavelengths
    pixel_coords_np = define_pixel_coordinates(GEOMETRY)
    if pixel_coords_np.shape[0] == 0:
        logging.error("No pixel coordinates defined. Exiting.")
        return
    num_pixels = pixel_coords_np.shape[0]
    lambdas_np, weights_np = sample_beam_wavelengths(BEAM_PARAMS, BEAM_PARAMS['n_lambda_samples'])
    if lambdas_np.size == 0:
        logging.error("Wavelength sampling failed. Exiting.")
        return
    weights = torch.tensor(weights_np, dtype=DTYPE_REAL, device=DEVICE)

    # 2. Generate q-vectors (Ideal and Sampled)
    logging.info("Generating q-vectors...")
    q_ideal_np = pixel_to_q(pixel_coords_np, GEOMETRY['wavelength_avg_a'], GEOMETRY)
    q_ideal_tensor = torch.tensor(q_ideal_np, dtype=DTYPE_REAL, device=DEVICE)

    q_samples_list = []
    pixel_lambda_to_q_map = {}
    q_vector_map = {}
    unique_q_indices_per_pixel = [[] for _ in range(num_pixels)]
    current_q_index = 0
    for i_lam, lam in enumerate(lambdas_np):
        q_for_lambda = pixel_to_q(pixel_coords_np, lam, GEOMETRY)
        for i_pix in range(num_pixels):
            q_vec = tuple(q_for_lambda[i_pix]) # Use tuple as dict key for precision
            if q_vec not in q_vector_map:
                q_vector_map[q_vec] = current_q_index
                q_samples_list.append(q_vec)
                q_index = current_q_index
                current_q_index += 1
            else:
                q_index = q_vector_map[q_vec]
            # Store mapping: which q_index corresponds to this pixel/lambda pair
            # pixel_lambda_to_q_map[(i_pix, i_lam)] = q_index # This map might not be needed
            unique_q_indices_per_pixel[i_pix].append(q_index)

    q_all_samples_np = np.array(q_samples_list, dtype=np.float64)
    q_all_samples_tensor = torch.tensor(q_all_samples_np, dtype=DTYPE_REAL, device=DEVICE)
    logging.info(f"Generated {q_ideal_tensor.shape[0]} ideal q-vectors (pixels).")
    logging.info(f"Generated {q_all_samples_tensor.shape[0]} unique q-vectors from wavelength sampling.")

    # --- 3. Generate Ground Truth Data ---
    logging.info("\n--- Generating Ground Truth Simulation (Realistic Beam, True Gamma) ---")
    I_pseudo_exp = None
    gt_generation_successful = False
    try:
        # Use non-optimizable tensors for true gamma
        gamma_true_intra_tensor = torch.tensor(GAMMA_TRUE_INTRA, dtype=DTYPE_REAL, device=DEVICE)
        gamma_true_inter_tensor = torch.tensor(GAMMA_TRUE_INTER, dtype=DTYPE_REAL, device=DEVICE)

        common_params_true_gamma = {
            **SIM_PARAMS,
            'gamma_intra': gamma_true_intra_tensor,
            'gamma_inter': gamma_true_inter_tensor,
            'device': DEVICE
        }

        # Simulate with all sampled q-vectors and TRUE gamma
        # Use detached q-vectors as we don't need gradients w.r.t. q here
        model_ground_truth = OnePhonon(
            pdb_path=PDB_PATH,
            q_vectors=q_all_samples_tensor.clone().detach(),
            **common_params_true_gamma
        )
        with torch.no_grad(): # No gradients needed for ground truth generation
            I_sim_samples_true = model_ground_truth.apply_disorder()
            logging.info(f"Ground truth simulation completed. Intensity shape: {I_sim_samples_true.shape}")

        # Average the ground truth simulation per pixel
        logging.info("Averaging ground truth simulation over wavelengths...")
        I_true_avg = torch.zeros(num_pixels, dtype=DTYPE_REAL, device=DEVICE)
        for i_pix in range(num_pixels):
            # Indices might be duplicated if multiple lambdas map to same q, which is fine
            q_indices_for_pixel = torch.tensor(unique_q_indices_per_pixel[i_pix], dtype=torch.long, device=DEVICE)
            # Ensure indices are within bounds
            if torch.any(q_indices_for_pixel >= I_sim_samples_true.shape[0]):
                logging.error(f"Index out of bounds for pixel {i_pix}. Max index: {q_indices_for_pixel.max()}, Sim shape: {I_sim_samples_true.shape}")
                I_true_avg[i_pix] = torch.tensor(float('nan'), device=DEVICE) # Mark as invalid
                continue
            sim_intensities_for_pixel = I_sim_samples_true[q_indices_for_pixel]

            # Check for NaNs in simulated data before averaging
            if torch.isnan(sim_intensities_for_pixel).any():
                 logging.warning(f"NaN found in simulated intensity for pixel {i_pix}. Setting average to NaN.")
                 I_true_avg[i_pix] = torch.tensor(float('nan'), device=DEVICE)
                 continue

            # Perform weighted average using the lambda weights
            # Assumes the order in unique_q_indices_per_pixel corresponds to the order of lambdas
            if len(unique_q_indices_per_pixel[i_pix]) != len(weights):
                 logging.error(f"Mismatch between number of q-indices ({len(unique_q_indices_per_pixel[i_pix])}) and weights ({len(weights)}) for pixel {i_pix}.")
                 I_true_avg[i_pix] = torch.tensor(float('nan'), device=DEVICE)
                 continue
            I_true_avg[i_pix] = torch.sum(weights * sim_intensities_for_pixel)
        logging.info("Averaging complete.")

        # Add noise to create pseudo-experimental data
        I_pseudo_exp = simulate_noisy_measurement(I_true_avg, MEAN_PHOTONS_PER_PIXEL)
        logging.info(f"Generated noisy pseudo-experimental data. Shape: {I_pseudo_exp.shape}")
        gt_generation_successful = True

        # --- Cleanup ---
        del model_ground_truth, I_sim_samples_true, I_true_avg
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Failed to generate ground truth data: {e}", exc_info=True)

    if not gt_generation_successful or I_pseudo_exp is None:
         logging.error("Pseudo-experimental data generation failed. Exiting.")
         return

    # --- 4. Setup for Gradient Calculation ---
    # Use INITIAL gamma values for the tensors that require gradients
    gamma_intra_tensor = nn.Parameter(torch.tensor(INITIAL_GAMMA_INTRA, dtype=DTYPE_REAL, device=DEVICE))
    gamma_inter_tensor = nn.Parameter(torch.tensor(INITIAL_GAMMA_INTER, dtype=DTYPE_REAL, device=DEVICE))

    loss_fn = nn.MSELoss()

    # Combine simulation parameters with the *optimizable* gamma tensors
    common_params_optimizable_gamma = {
        **SIM_PARAMS,
        'gamma_intra': gamma_intra_tensor,
        'gamma_inter': gamma_inter_tensor,
        'device': DEVICE
    }

    # --- 5. Scenario A: Fit with Ideal Beam Model ---
    logging.info("\n--- Running Scenario A: Fit with Ideal Beam Model ---")
    grad_gamma_intra_ideal = torch.tensor(float('nan'), device=DEVICE)
    grad_gamma_inter_ideal = torch.tensor(float('nan'), device=DEVICE)
    loss_a_val = float('nan')
    scenario_a_successful = False
    try:
        # Ensure q_ideal requires grad only if optimizing q itself
        q_ideal_tensor_fit = q_ideal_tensor.clone().detach().requires_grad_(False)

        model_ideal = OnePhonon(
            pdb_path=PDB_PATH,
            q_vectors=q_ideal_tensor_fit,
            **common_params_optimizable_gamma # Use optimizable gamma
        )

        I_sim_ideal = model_ideal.apply_disorder()
        logging.info(f"Scenario A simulation completed. Intensity shape: {I_sim_ideal.shape}")

        # Calculate loss against pseudo-experimental data
        valid_mask_ideal = ~torch.isnan(I_sim_ideal) & ~torch.isnan(I_pseudo_exp)
        if not torch.any(valid_mask_ideal):
             logging.warning("Scenario A: No valid overlapping data points for loss calculation.")
             raise ValueError("No valid points for loss calculation in Scenario A")

        loss_ideal = loss_fn(I_sim_ideal[valid_mask_ideal], I_pseudo_exp[valid_mask_ideal])
        loss_a_val = loss_ideal.item()

        # Compute gradients
        gamma_intra_tensor.grad = None # Zero grads before backward
        gamma_inter_tensor.grad = None
        loss_ideal.backward()

        # Store gradients safely
        grad_gamma_intra_ideal = gamma_intra_tensor.grad.clone() if gamma_intra_tensor.grad is not None else torch.zeros_like(gamma_intra_tensor)
        grad_gamma_inter_ideal = gamma_inter_tensor.grad.clone() if gamma_inter_tensor.grad is not None else torch.zeros_like(gamma_inter_tensor)

        logging.info(f"Scenario A Loss: {loss_a_val:.6e}")
        logging.info(f"Scenario A Grad (Intra): {grad_gamma_intra_ideal.item():.6e}")
        logging.info(f"Scenario A Grad (Inter): {grad_gamma_inter_ideal.item():.6e}")
        scenario_a_successful = True

        # --- Cleanup ---
        del model_ideal, I_sim_ideal, loss_ideal, q_ideal_tensor_fit
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Error in Scenario A: {e}", exc_info=True)


    # --- 6. Scenario B: Fit with Realistic Beam Model ---
    logging.info("\n--- Running Scenario B: Fit with Realistic Beam Model ---")
    grad_gamma_intra_realistic = torch.tensor(float('nan'), device=DEVICE)
    grad_gamma_inter_realistic = torch.tensor(float('nan'), device=DEVICE)
    loss_b_val = float('nan')
    scenario_b_successful = False
    try:
        # Ensure q_all_samples requires grad only if optimizing q
        q_all_samples_tensor_fit = q_all_samples_tensor.clone().detach().requires_grad_(False)

        model_realistic = OnePhonon(
            pdb_path=PDB_PATH,
            q_vectors=q_all_samples_tensor_fit,
            **common_params_optimizable_gamma # Use optimizable gamma
        )

        I_sim_samples = model_realistic.apply_disorder()
        logging.info(f"Scenario B simulation completed. Intensity shape: {I_sim_samples.shape}")

        # Average results per pixel
        logging.info("Averaging Scenario B simulation over wavelengths...")
        I_sim_avg = torch.zeros_like(I_pseudo_exp)
        for i_pix in range(num_pixels):
            q_indices_for_pixel = torch.tensor(unique_q_indices_per_pixel[i_pix], dtype=torch.long, device=DEVICE)
            # Ensure indices are within bounds
            if torch.any(q_indices_for_pixel >= I_sim_samples.shape[0]):
                 logging.error(f"Index out of bounds for pixel {i_pix} in Scenario B. Max index: {q_indices_for_pixel.max()}, Sim shape: {I_sim_samples.shape}")
                 I_sim_avg[i_pix] = torch.tensor(float('nan'), device=DEVICE)
                 continue
            sim_intensities_for_pixel = I_sim_samples[q_indices_for_pixel]

            # Check for NaNs before averaging
            if torch.isnan(sim_intensities_for_pixel).any():
                 logging.warning(f"NaN found in simulated intensity for pixel {i_pix} (Scenario B). Setting average to NaN.")
                 I_sim_avg[i_pix] = torch.tensor(float('nan'), device=DEVICE)
                 continue
            # Perform weighted average
            if len(unique_q_indices_per_pixel[i_pix]) != len(weights):
                 logging.error(f"Mismatch between number of q-indices ({len(unique_q_indices_per_pixel[i_pix])}) and weights ({len(weights)}) for pixel {i_pix} (Scenario B).")
                 I_sim_avg[i_pix] = torch.tensor(float('nan'), device=DEVICE)
                 continue
            I_sim_avg[i_pix] = torch.sum(weights * sim_intensities_for_pixel)
        logging.info("Averaging complete.")

        # Calculate loss against pseudo-experimental data
        valid_mask_realistic = ~torch.isnan(I_sim_avg) & ~torch.isnan(I_pseudo_exp)
        if not torch.any(valid_mask_realistic):
             logging.warning("Scenario B: No valid overlapping data points for loss calculation.")
             raise ValueError("No valid points for loss calculation in Scenario B")

        loss_realistic = loss_fn(I_sim_avg[valid_mask_realistic], I_pseudo_exp[valid_mask_realistic])
        loss_b_val = loss_realistic.item()

        # Compute gradients
        gamma_intra_tensor.grad = None # Zero grads before backward
        gamma_inter_tensor.grad = None
        loss_realistic.backward()

        # Store gradients safely
        grad_gamma_intra_realistic = gamma_intra_tensor.grad.clone() if gamma_intra_tensor.grad is not None else torch.zeros_like(gamma_intra_tensor)
        grad_gamma_inter_realistic = gamma_inter_tensor.grad.clone() if gamma_inter_tensor.grad is not None else torch.zeros_like(gamma_inter_tensor)

        logging.info(f"Scenario B Loss: {loss_b_val:.6e}")
        logging.info(f"Scenario B Grad (Intra): {grad_gamma_intra_realistic.item():.6e}")
        logging.info(f"Scenario B Grad (Inter): {grad_gamma_inter_realistic.item():.6e}")
        scenario_b_successful = True

        # --- Cleanup ---
        del model_realistic, I_sim_samples, I_sim_avg, loss_realistic, q_all_samples_tensor_fit
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Error in Scenario B: {e}", exc_info=True)


    # --- 7. Compare Gradients & Interpret ---
    logging.info("\n--- Gradient Comparison & Interpretation ---")

    if not scenario_a_successful or not scenario_b_successful:
        logging.error("Cannot compare gradients because one or both scenarios failed.")
    elif torch.isnan(grad_gamma_intra_ideal).any() or torch.isnan(grad_gamma_inter_ideal).any() or \
         torch.isnan(grad_gamma_intra_realistic).any() or torch.isnan(grad_gamma_inter_realistic).any():
        logging.warning("Could not compare gradients due to NaN values computed in one or both scenarios.")
    else:
        # Combine gradients into vectors for comparison
        grad_vec_ideal = torch.stack([grad_gamma_intra_ideal, grad_gamma_inter_ideal])
        grad_vec_realistic = torch.stack([grad_gamma_intra_realistic, grad_gamma_inter_realistic])

        norm_ideal = torch.norm(grad_vec_ideal)
        norm_realistic = torch.norm(grad_vec_realistic)
        eps = 1e-10 # Small epsilon for safe division

        # Calculate metrics only if norms are reasonably large
        if norm_ideal < eps or norm_realistic < eps:
             logging.warning(f"One or both gradients have near-zero norm (Ideal: {norm_ideal:.2e}, Realistic: {norm_realistic:.2e}). Comparison might be unreliable.")
             cos_sim = torch.tensor(float('nan'), device=DEVICE)
             rel_diff = torch.norm(grad_vec_ideal - grad_vec_realistic) # Absolute difference if norm is zero
        else:
            cos_sim = torch.dot(grad_vec_ideal, grad_vec_realistic) / (norm_ideal * norm_realistic)
            rel_diff = torch.norm(grad_vec_ideal - grad_vec_realistic) / norm_ideal

        intra_abs_diff = torch.abs(grad_gamma_intra_ideal - grad_gamma_intra_realistic)
        inter_abs_diff = torch.abs(grad_gamma_inter_ideal - grad_gamma_inter_realistic)
        intra_rel_diff = intra_abs_diff / (torch.abs(grad_gamma_intra_ideal) + eps)
        inter_rel_diff = inter_abs_diff / (torch.abs(grad_gamma_inter_ideal) + eps)

        logging.info(f"Cosine Similarity: {cos_sim.item():.4f}")
        logging.info(f"Relative Norm Difference: {rel_diff.item():.4f}")
        logging.info(f"Intra Grad Abs/Rel Diff: {intra_abs_diff.item():.6e} / {intra_rel_diff.item():.4f}")
        logging.info(f"Inter Grad Abs/Rel Diff: {inter_abs_diff.item():.6e} / {inter_rel_diff.item():.4f}")

        # Interpretation logic
        is_similar_direction = cos_sim.item() > COS_SIM_THRESHOLD
        is_similar_magnitude = rel_diff.item() < REL_DIFF_THRESHOLD

        if torch.isnan(cos_sim):
             logging.warning("Interpretation skipped due to NaN cosine similarity (likely zero gradients).")
        elif is_similar_direction and is_similar_magnitude:
            logging.info("Conclusion: Gradients are similar. Beam jitter/bandwidth effect on gamma estimation is likely NEGLIGIBLE for the assumed noise level and beam parameters.")
        elif is_similar_direction:
            logging.warning("Conclusion: Gradient directions are similar, but magnitudes differ significantly. Beam effects might primarily scale optimization steps. Effect is potentially MINOR but warrants caution.")
        else:
            logging.warning("Conclusion: Gradients differ significantly in direction. Beam jitter/bandwidth effect is likely SIGNIFICANT and should ideally be included in the model for accurate gamma estimation, even with this noise level.")

    total_time = time.time() - start_time
    logging.info(f"\nAnalysis finished in {total_time:.2f} seconds.")
    logging.info("="*60)


# --- Script Execution ---
if __name__ == "__main__":
    run_analysis()
