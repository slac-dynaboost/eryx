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
# Provide sampling params EVEN for arbitrary mode for consistent ADP calc
hsampling = [-1, 1, 2]
ksampling = [-1, 1, 2]
lsampling = [-1, 1, 2]
device = torch.device('cpu')
common_params = { 'expand_p1': True, 'group_by': 'asu', 'res_limit': 0.0, 'model': 'gnm', 'gnm_cutoff': 4.0, 'gamma_intra': 1.0, 'gamma_inter': 1.0, 'n_processes': 1 }

# Load the q_grid saved from the grid run
q_grid_file = "phase0_q_grid.npy"
if not os.path.exists(q_grid_file):
    print(f"Error: {q_grid_file} not found. Please run the grid script first.")
    sys.exit(1)

q_vectors_from_grid = torch.from_numpy(np.load(q_grid_file)).to(device)
print(f"Loaded q_vectors shape: {q_vectors_from_grid.shape} from {q_grid_file}")

print("\nRunning Arbitrary-Q Mode...")
model_arb = OnePhononTorch(
    pdb_path=pdb_path,
    q_vectors=q_vectors_from_grid, # Use saved q_grid
    hsampling=hsampling,          # Still provide sampling for ADP calc
    ksampling=ksampling,
    lsampling=lsampling,
    device=device,
    **common_params
)

# Run apply_disorder
print("\nCalling apply_disorder for Arbitrary-Q Mode...")
intensity_arb = model_arb.apply_disorder(use_data_adp=True)
print(f"Arbitrary-Q mode intensity calculated, shape: {intensity_arb.shape}")
print("Arbitrary-Q Mode script finished.")
