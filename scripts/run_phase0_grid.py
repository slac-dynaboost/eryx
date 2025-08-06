import torch
import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from eryx.models_torch import OnePhonon as OnePhononTorch

# Parameters from Step 1
pdb_path = 'tests/pdbs/5zck_p1.pdb'
hsampling = [-1, 1, 2]
ksampling = [-1, 1, 2]
lsampling = [-1, 1, 2]
device = torch.device('cpu')
common_params = { 'expand_p1': True, 'group_by': 'asu', 'res_limit': 0.0, 'model': 'gnm', 'gnm_cutoff': 4.0, 'gamma_intra': 1.0, 'gamma_inter': 1.0, 'n_processes': 1 }

print("Running Grid Mode...")
model_grid = OnePhononTorch(
    pdb_path=pdb_path,
    hsampling=hsampling,
    ksampling=ksampling,
    lsampling=lsampling,
    device=device,
    **common_params
)

# Save the q_grid for the arbitrary mode run
q_grid_for_arb = model_grid.q_grid.clone().detach()
np.save("phase0_q_grid.npy", q_grid_for_arb.cpu().numpy())
print(f"Saved q_grid shape: {q_grid_for_arb.shape} to phase0_q_grid.npy")

# --- Find target_q_idx ---
# Add this temporarily to find target_q_idx
target_dh, target_dk, target_dl = 0, 0, 1
# Ensure BZ indices are tensors for _flat_to_3d_indices_bz if needed, or use the tuple directly
# Assuming _at_kvec_from_miller_points takes a tuple (dh, dk, dl)
q_indices_for_target_bz = model_grid._at_kvec_from_miller_points((target_dh, target_dk, target_dl))

if q_indices_for_target_bz is not None and q_indices_for_target_bz.numel() > 0:
    target_q_idx = q_indices_for_target_bz[0].item()
    print(f"*** Target q_index for BZ({target_dh},{target_dk},{target_dl}) is: {target_q_idx} ***")
else:
    print(f"*** ERROR: No q_indices found for BZ({target_dh},{target_dk},{target_dl}) ***")
    target_q_idx = -1 # Indicate error
# --- End Find target_q_idx ---


# Run apply_disorder to ensure it completes without error and generate logs
print("\nCalling apply_disorder for Grid Mode...")
intensity_grid = model_grid.apply_disorder(use_data_adp=True)
print(f"Grid mode intensity calculated, shape: {intensity_grid.shape}")
print("Grid Mode script finished.")
