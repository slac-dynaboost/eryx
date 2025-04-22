import torch
import numpy as np
import os
import logging
import time
from typing import Tuple, Dict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()] # Log to console for simplicity
)

# -- Files and Paths --
# !!! REPLACE WITH YOUR ACTUAL PDB PATH !!!
PDB_PATH = "tests/pdbs/5zck_p1.pdb" # Using test PDB for example run

# -- Simulation Parameters --
# !!! ADJUST THESE FOR YOUR SYSTEM !!!
SIM_PARAMS = {
    'expand_p1': True,
    'group_by': 'asu',
    'model': 'gnm',
    'gnm_cutoff': 7.0,
    # Sampling params needed even for arbitrary-q if using computed ADP
    'hsampling': [-4, 4, 3],
    'ksampling': [-4, 4, 3],
    'lsampling': [-4, 4, 3],
    # 'use_data_adp' removed - should be passed to apply_disorder if needed
}
# Use fixed gamma values for this estimation
GAMMA_INTRA = 1.5
GAMMA_INTER = 0.7

# -- Back-of-Envelope Parameters --
TARGET_RESOLUTION_A = 2.0 # Target resolution to probe (Angstrom)
DQ_FRAC = 0.001           # Fractional step size for finite difference (0.1%)
REL_ENERGY_JITTER = 0.001 # Assumed relative energy jitter Δω/ω (e.g., 0.1%)
MEAN_PHOTONS_PER_PIXEL = 1e8 # !!! ADJUST: Estimated mean photons per pixel for noise level !!!

# -- Computation Settings --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE_REAL = torch.float64
DTYPE_COMPLEX = torch.complex128

# --- Import Eryx Components ---
try:
    from eryx.models_torch import OnePhonon
except ImportError as e:
    logging.error(f"Could not import Eryx components. Error: {e}")
    exit()
except Exception as e:
     logging.error(f"An unexpected error occurred during Eryx import: {e}")
     exit()

# --- Main Calculation ---

def run_back_of_envelope():
    """Performs the back-of-the-envelope calculation."""
    start_time = time.time()
    logging.info("="*50)
    logging.info("Back-of-Envelope: Diffuse Intensity Sensitivity to Jitter")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Using PDB: {PDB_PATH}")
    if not os.path.exists(PDB_PATH):
         logging.error(f"PDB file not found at {PDB_PATH}. Exiting.")
         return

    # 1. Setup Model (once)
    logging.info("Setting up OnePhonon model...")
    try:
        # Use fixed gamma values (no gradients needed for gamma here)
        gamma_intra_tensor = torch.tensor(GAMMA_INTRA, dtype=DTYPE_REAL, device=DEVICE)
        gamma_inter_tensor = torch.tensor(GAMMA_INTER, dtype=DTYPE_REAL, device=DEVICE)

        common_params = {
            **SIM_PARAMS,
            'gamma_intra': gamma_intra_tensor,
            'gamma_inter': gamma_inter_tensor,
            'device': DEVICE
        }
        # Instantiate model - will compute phonons etc.
        # We need a dummy q_vector initially, will replace later
        dummy_q = torch.zeros((1, 3), dtype=DTYPE_REAL, device=DEVICE)
        model = OnePhonon(
            pdb_path=PDB_PATH,
            q_vectors=dummy_q, # Provide dummy q_vectors
            **common_params
        )
        logging.info("Model setup complete.")

    except Exception as e:
        logging.error(f"Failed to initialize OnePhonon model: {e}", exc_info=True)
        return

    # 2. Choose Representative q-point
    q_mag = 2.0 * np.pi / TARGET_RESOLUTION_A
    # Choose an arbitrary direction, e.g., along x-axis
    q0_vec_np = np.array([q_mag, 0.0, 0.0], dtype=np.float64)
    q0_vec = torch.tensor(q0_vec_np, dtype=DTYPE_REAL, device=DEVICE).unsqueeze(0) # Shape (1, 3)
    logging.info(f"Probing at |q| = {q_mag:.3f} Å⁻¹ (Resolution = {TARGET_RESOLUTION_A:.2f} Å)")
    logging.info(f"Using q0 = {q0_vec.cpu().numpy()}")

    # 3. Define Perturbed q-point (radial step)
    dq_mag = q_mag * DQ_FRAC
    q1_vec = q0_vec + dq_mag * (q0_vec / (q_mag + 1e-9)) # Add small epsilon for stability
    logging.info(f"Using q1 = {q1_vec.cpu().numpy()} (|q1| = {torch.norm(q1_vec).item():.3f} Å⁻¹)")

    # Combine q-vectors for batch simulation
    q_eval = torch.cat([q0_vec, q1_vec], dim=0) # Shape (2, 3)

    # 4. Simulate Intensity at q0 and q1
    logging.info("Simulating intensities at q0 and q1...")
    try:
        # Temporarily replace model's q_grid and related attributes
        # This is a bit hacky, ideally the model would take q directly in apply_disorder
        # Or we re-initialize, but that's slow. Let's try modifying state carefully.
        original_q_grid = model.q_grid
        original_hkl_grid = model.hkl_grid
        original_map_shape = model.map_shape
        original_res_mask = model.res_mask
        original_kvec = model.kvec # Need to recalculate kvec for new q!
        original_kvec_norm = model.kvec_norm
        original_V = model.V
        original_Winv = model.Winv

        # Set arbitrary mode flag (might be needed by internal checks)
        model.use_arbitrary_q = True
        model.q_grid = q_eval.clone().detach().requires_grad_(False) # No grad needed for q here
        model.map_shape = (2, 1, 1) # Dummy map shape for 2 points
        model.res_mask = torch.ones(2, dtype=torch.bool, device=DEVICE) # Assume points are valid

        # --- Recalculate attributes dependent on q_grid ---
        # Recalculate kvec based on the new q_grid
        model._build_kvec_Brillouin() # This recalculates kvec and kvec_norm based on current q_grid
        logging.info(f"Recalculated kvec for eval points. Shape: {model.kvec.shape}")

        # Recalculate phonons (V, Winv) for the new k-vectors
        # This is the expensive step, but necessary if V/Winv depend strongly on kvec
        logging.info("Recalculating phonons for evaluation points...")
        model.compute_gnm_phonons() # This recalculates V and Winv based on current kvec
        logging.info("Phonon recalculation complete.")
        # --- End Recalculation ---


        with torch.no_grad(): # No gradients needed for this estimation
            I_d_eval = model.apply_disorder() # Shape (2,)
            logging.info(f"Simulation complete. Intensity shape: {I_d_eval.shape}")

        # Restore original model state
        model.q_grid = original_q_grid
        model.hkl_grid = original_hkl_grid
        model.map_shape = original_map_shape
        model.res_mask = original_res_mask
        model.kvec = original_kvec
        model.kvec_norm = original_kvec_norm
        model.V = original_V
        model.Winv = original_Winv
        model.use_arbitrary_q = False # Reset flag


        I0 = I_d_eval[0].item()
        I1 = I_d_eval[1].item()

        # Handle potential NaN or zero intensities
        if np.isnan(I0) or np.isnan(I1) or I0 <= 1e-12:
            logging.warning(f"Simulation yielded NaN or near-zero intensity (I0={I0:.4e}, I1={I1:.4e}). Cannot estimate derivative.")
            return

        logging.info(f"I_d(q0) = {I0:.4e}")
        logging.info(f"I_d(q1) = {I1:.4e}")

    except Exception as e:
        logging.error(f"Failed during simulation: {e}", exc_info=True)
        return

    # 5. Estimate Logarithmic Derivative
    dId = I1 - I0
    dq = dq_mag # Magnitude of the step
    dId_dq = dId / dq
    dlnId_dq = (1.0 / I0) * dId_dq

    logging.info(f"Estimated d(ln I_d)/d|q| ≈ {dlnId_dq:.4e} Å")

    # 6. Calculate Fractional Change
    frac_change = abs(dlnId_dq) * REL_ENERGY_JITTER * q_mag

    logging.info(f"Assumed Δω/ω = {REL_ENERGY_JITTER:.4f}")
    logging.info(f"Estimated |∇_q ln I_d ⋅ q/|q|| ≈ {abs(dlnId_dq):.4e} Å")
    logging.info(f"Resulting Fractional Change ΔI_d / I_d ≈ {frac_change:.4f}")

    # 7. Interpretation
    logging.info("\n--- Interpretation ---")
    noise_level_frac = 1.0 / np.sqrt(MEAN_PHOTONS_PER_PIXEL) if MEAN_PHOTONS_PER_PIXEL > 0 else float('inf')
    logging.info(f"Assumed fractional Poisson noise level (1/√N) ≈ {noise_level_frac:.4f} (for {MEAN_PHOTONS_PER_PIXEL} photons)")

    if frac_change < 0.1 * noise_level_frac:
        logging.info("Conclusion: Estimated intensity change due to jitter is MUCH SMALLER than Poisson noise.")
        logging.info(">> Beam jitter/bandwidth effect is likely NEGLIGIBLE for gamma estimation.")
    elif frac_change < noise_level_frac:
        logging.info("Conclusion: Estimated intensity change due to jitter is SMALLER than Poisson noise.")
        logging.info(">> Beam jitter/bandwidth effect is likely MINOR for gamma estimation.")
    elif frac_change < 5 * noise_level_frac:
        logging.info("Conclusion: Estimated intensity change due to jitter is COMPARABLE to Poisson noise.")
        logging.info(">> Beam jitter/bandwidth effect MAY BE SIGNIFICANT. Inclusion in the model is recommended for high accuracy.")
    else:
        logging.info("Conclusion: Estimated intensity change due to jitter is LARGER than Poisson noise.")
        logging.info(">> Beam jitter/bandwidth effect is likely SIGNIFICANT and should be included in the model.")

    total_time = time.time() - start_time
    logging.info(f"\nCalculation finished in {total_time:.2f} seconds.")
    logging.info("="*50)


# --- Script Execution ---
if __name__ == "__main__":
    run_back_of_envelope()
